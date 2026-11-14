#ifndef TRANSFORMER_LAYER_HPP
#define TRANSFORMER_LAYER_HPP

#include <cstddef>
#include <vector>
#include <memory>
#include "gemv.hpp"

namespace streamvex {

struct TransformerConfig {
    int dim = 512;
    int hidden_dim = 1376;
    int n_heads = 8;
    int n_kv_heads = 8;
    int head_dim = 64; // dim / n_heads
    int max_seq_len = 2048;
    int vocab_size = 4096;
    float norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
};

/**
 * Compact Key-Value Cache capped in RAM.
 * Memory footprint: max_seq_len * n_kv_heads * head_dim * sizeof(float) * 2
 */
class CompactKVCache {
public:
    CompactKVCache(int max_seq_len, int n_kv_heads, int head_dim);

    float* get_k_pos(int pos) { return m_k.data() + pos * m_stride; }
    float* get_v_pos(int pos) { return m_v.data() + pos * m_stride; }

    const float* get_k_all() const { return m_k.data(); }
    const float* get_v_all() const { return m_v.data(); }

    int max_seq_len() const { return m_max_seq_len; }
    int n_kv_heads() const { return m_n_kv_heads; }
    int head_dim() const { return m_head_dim; }
    int stride() const { return m_stride; }
    size_t memory_bytes() const { return (m_k.size() + m_v.size()) * sizeof(float); }

    void reset() {
        std::fill(m_k.begin(), m_k.end(), 0.0f);
        std::fill(m_v.begin(), m_v.end(), 0.0f);
    }

private:
    int m_max_seq_len;
    int m_n_kv_heads;
    int m_head_dim;
    int m_stride;
    std::vector<float> m_k;
    std::vector<float> m_v;
};

/**
 * Single Transformer Layer Weights (stored in Q4_0 quantized format or float32).
 */
struct LayerWeights {
    std::vector<float> rms_att_weight;
    std::vector<block_q4_0> wq_q4;
    std::vector<block_q4_0> wk_q4;
    std::vector<block_q4_0> wv_q4;
    std::vector<block_q4_0> wo_q4;

    std::vector<float> rms_ffn_weight;
    std::vector<block_q4_0> w_gate_q4;
    std::vector<block_q4_0> w_up_q4;
    std::vector<block_q4_0> w_down_q4;

    // Helper to generate normalized synthetic weights for testing and benchmarking
    void init_synthetic(const TransformerConfig& cfg);
};

/**
 * Root Mean Square Normalization: out = (x / rms(x)) * weight
 */
void rmsnorm(const float* x, const float* weight, float* out, int dim, float eps = 1e-5f);

/**
 * Rotary Positional Embeddings (RoPE) applied to query and key vectors.
 */
void apply_rope(
    float* q,
    float* k,
    int pos,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float theta = 10000.0f
);

/**
 * Executes the complete forward pass pipeline for a single Transformer layer block:
 * 1. RMSNorm
 * 2. QKV Projection (via GEMV)
 * 3. RoPE
 * 4. Multi-Head Attention with KV-Cache
 * 5. Output Projection + Residual Add
 * 6. RMSNorm
 * 7. SwiGLU FFN (Gate + Up + Swish + Down) + Residual Add
 */
void transformer_layer_forward(
    const TransformerConfig& cfg,
    const LayerWeights& weights,
    CompactKVCache& kv_cache,
    int pos,
    float* x, // Input activation, updated in-place with residual
    float* scratchpad // Temporary workspace buffer
);

} // namespace streamvex

#endif // TRANSFORMER_LAYER_HPP
