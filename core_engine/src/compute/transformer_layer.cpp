#include "transformer_layer.hpp"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <iostream>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace streamvex {

CompactKVCache::CompactKVCache(int max_seq_len, int n_kv_heads, int head_dim)
    : m_max_seq_len(max_seq_len),
      m_n_kv_heads(n_kv_heads),
      m_head_dim(head_dim),
      m_stride(n_kv_heads * head_dim) {
    m_k.resize(static_cast<size_t>(m_max_seq_len) * m_stride, 0.0f);
    m_v.resize(static_cast<size_t>(m_max_seq_len) * m_stride, 0.0f);
}

void rmsnorm(const float* x, const float* weight, float* out, int dim, float eps) {
    float sum_sq = 0.0f;

#if defined(__ARM_NEON)
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i <= dim - 16; i += 16) {
        float32x4_t x0 = vld1q_f32(x + i + 0);
        float32x4_t x1 = vld1q_f32(x + i + 4);
        float32x4_t x2 = vld1q_f32(x + i + 8);
        float32x4_t x3 = vld1q_f32(x + i + 12);

        sum_vec = vmlaq_f32(sum_vec, x0, x0);
        sum_vec = vmlaq_f32(sum_vec, x1, x1);
        sum_vec = vmlaq_f32(sum_vec, x2, x2);
        sum_vec = vmlaq_f32(sum_vec, x3, x3);
    }
    sum_sq = vaddvq_f32(sum_vec);
    for (; i < dim; ++i) {
        sum_sq += x[i] * x[i];
    }
#else
    for (int i = 0; i < dim; ++i) {
        sum_sq += x[i] * x[i];
    }
#endif

    float inv_rms = 1.0f / std::sqrt((sum_sq / static_cast<float>(dim)) + eps);

#if defined(__ARM_NEON)
    float32x4_t scale_vec = vdupq_n_f32(inv_rms);
    int j = 0;
    for (; j <= dim - 4; j += 4) {
        float32x4_t xv = vld1q_f32(x + j);
        float32x4_t wv = vld1q_f32(weight + j);
        float32x4_t res = vmulq_f32(vmulq_f32(xv, scale_vec), wv);
        vst1q_f32(out + j, res);
    }
    for (; j < dim; ++j) {
        out[j] = x[j] * inv_rms * weight[j];
    }
#else
    for (int j = 0; j < dim; ++j) {
        out[j] = x[j] * inv_rms * weight[j];
    }
#endif
}

void apply_rope(
    float* q,
    float* k,
    int pos,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float theta
) {
    for (int h = 0; h < n_heads; ++h) {
        float* q_head = q + h * head_dim;
        for (int i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / std::pow(theta, static_cast<float>(i) / static_cast<float>(head_dim));
            float val = static_cast<float>(pos) * freq;
            float fcr = std::cos(val);
            float fci = std::sin(val);

            float q0 = q_head[i];
            float q1 = q_head[i + 1];
            q_head[i]     = q0 * fcr - q1 * fci;
            q_head[i + 1] = q0 * fci + q1 * fcr;
        }
    }

    for (int h = 0; h < n_kv_heads; ++h) {
        float* k_head = k + h * head_dim;
        for (int i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / std::pow(theta, static_cast<float>(i) / static_cast<float>(head_dim));
            float val = static_cast<float>(pos) * freq;
            float fcr = std::cos(val);
            float fci = std::sin(val);

            float k0 = k_head[i];
            float k1 = k_head[i + 1];
            k_head[i]     = k0 * fcr - k1 * fci;
            k_head[i + 1] = k0 * fci + k1 * fcr;
        }
    }
}

void LayerWeights::init_synthetic(const TransformerConfig& cfg) {
    int dim = cfg.dim;
    int hidden_dim = cfg.hidden_dim;

    rms_att_weight.assign(dim, 1.0f);
    rms_ffn_weight.assign(dim, 1.0f);

    auto make_q4 = [](int rows, int cols) {
        std::vector<float> f_data(rows * cols);
        float scale = 1.0f / std::sqrt(static_cast<float>(cols));
        for (size_t i = 0; i < f_data.size(); ++i) {
            f_data[i] = std::sin(static_cast<float>(i) * 0.02f) * scale;
        }
        std::vector<block_q4_0> q_data((rows * cols) / 32);
        quantize_row_q4_0(f_data.data(), q_data.data(), rows * cols);
        return q_data;
    };

    wq_q4 = make_q4(dim, dim);
    wk_q4 = make_q4(dim, dim);
    wv_q4 = make_q4(dim, dim);
    wo_q4 = make_q4(dim, dim);

    w_gate_q4 = make_q4(hidden_dim, dim);
    w_up_q4   = make_q4(hidden_dim, dim);
    w_down_q4 = make_q4(dim, hidden_dim);
}

void transformer_layer_forward(
    const TransformerConfig& cfg,
    const LayerWeights& weights,
    CompactKVCache& kv_cache,
    int pos,
    float* x,
    float* scratchpad
) {
    const int dim = cfg.dim;
    const int hidden_dim = cfg.hidden_dim;
    const int n_heads = cfg.n_heads;
    const int n_kv_heads = cfg.n_kv_heads;
    const int head_dim = cfg.head_dim;

    // Scratchpad buffer partition
    float* x_norm = scratchpad;                           // size: dim
    float* q = x_norm + dim;                              // size: dim
    float* k = q + dim;                                   // size: dim
    float* v = k + dim;                                   // size: dim
    float* attn_out = v + dim;                            // size: dim
    float* gate = attn_out + dim;                         // size: hidden_dim
    float* up = gate + hidden_dim;                        // size: hidden_dim
    float* scores = up + hidden_dim;                      // size: max_seq_len

    // 1. Pre-Attention RMSNorm
    rmsnorm(x, weights.rms_att_weight.data(), x_norm, dim, cfg.norm_eps);

    // 2. QKV Projection via Quantized GEMV
    gemv_q4_0_f32(weights.wq_q4.data(), x_norm, q, dim, dim);
    gemv_q4_0_f32(weights.wk_q4.data(), x_norm, k, dim, dim);
    gemv_q4_0_f32(weights.wv_q4.data(), x_norm, v, dim, dim);

    // 3. Rotary Positional Embeddings (RoPE)
    apply_rope(q, k, pos, n_heads, n_kv_heads, head_dim, cfg.rope_theta);

    // 4. Update KV-Cache at current position
    float* k_cache_pos = kv_cache.get_k_pos(pos);
    float* v_cache_pos = kv_cache.get_v_pos(pos);
    std::memcpy(k_cache_pos, k, dim * sizeof(float));
    std::memcpy(v_cache_pos, v, dim * sizeof(float));

    // 5. Multi-Head Self-Attention over past tokens [0 .. pos]
    const float scale_att = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const float* all_k = kv_cache.get_k_all();
    const float* all_v = kv_cache.get_v_all();
    const int kv_stride = kv_cache.stride();

    for (int h = 0; h < n_heads; ++h) {
        const float* q_h = q + h * head_dim;
        float* out_h = attn_out + h * head_dim;

        // Compute attention scores with historical keys
        float max_score = -1e9f;
        for (int t = 0; t <= pos; ++t) {
            const float* k_t = all_k + t * kv_stride + h * head_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += q_h[d] * k_t[d];
            }
            score *= scale_att;
            scores[t] = score;
            if (score > max_score) max_score = score;
        }

        // Softmax
        float sum_exp = 0.0f;
        for (int t = 0; t <= pos; ++t) {
            scores[t] = std::exp(scores[t] - max_score);
            sum_exp += scores[t];
        }
        float inv_sum = 1.0f / (sum_exp + 1e-9f);

        // Weighted sum over historical values
        std::fill(out_h, out_h + head_dim, 0.0f);
        for (int t = 0; t <= pos; ++t) {
            float weight = scores[t] * inv_sum;
            const float* v_t = all_v + t * kv_stride + h * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                out_h[d] += weight * v_t[d];
            }
        }
    }

    // Output Projection: q is recycled as attn_proj output buffer
    float* attn_proj = q;
    gemv_q4_0_f32(weights.wo_q4.data(), attn_out, attn_proj, dim, dim);

    // Residual Connection: x = x + attn_proj
    for (int i = 0; i < dim; ++i) {
        x[i] += attn_proj[i];
    }

    // 6. Post-Attention RMSNorm
    rmsnorm(x, weights.rms_ffn_weight.data(), x_norm, dim, cfg.norm_eps);

    // 7. SwiGLU Feed-Forward Network (Gate & Up Projections)
    gemv_q4_0_f32(weights.w_gate_q4.data(), x_norm, gate, hidden_dim, dim);
    gemv_q4_0_f32(weights.w_up_q4.data(), x_norm, up, hidden_dim, dim);

    // Swish activation: gate = (gate * sigmoid(gate)) * up
    for (int i = 0; i < hidden_dim; ++i) {
        float g = gate[i];
        float silu = g / (1.0f + std::exp(-g));
        gate[i] = silu * up[i];
    }

    // Down Projection: ffn_out
    float* ffn_out = q;
    gemv_q4_0_f32(weights.w_down_q4.data(), gate, ffn_out, dim, hidden_dim);

    // Final Residual Connection: x = x + ffn_out
    for (int i = 0; i < dim; ++i) {
        x[i] += ffn_out[i];
    }
}

} // namespace streamvex
