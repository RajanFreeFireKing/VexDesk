#include "dequant.hpp"

#include <cmath>
#include <algorithm>
#include <iostream>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace streamvex {

#if defined(__ARM_NEON)
static void dequantize_q4_0_neon(const block_q4_0* b, float* y, size_t nb) {
    for (size_t i = 0; i < nb; ++i) {
        float d = fp16_to_fp32(b[i].d);
        float32x4_t vd = vdupq_n_f32(d);

        uint8x16_t q = vld1q_u8(b[i].qs);
        uint8x16_t q0 = vandq_u8(q, vdupq_n_u8(0x0F));
        uint8x16_t q1 = vshrq_n_u8(q, 4);

        int8x16_t s0 = vsubq_s8(vreinterpretq_s8_u8(q0), vdupq_n_s8(8));
        int8x16_t s1 = vsubq_s8(vreinterpretq_s8_u8(q1), vdupq_n_s8(8));

        int16x8_t s0_l = vmovl_s8(vget_low_s8(s0));
        int16x8_t s0_h = vmovl_s8(vget_high_s8(s0));
        int16x8_t s1_l = vmovl_s8(vget_low_s8(s1));
        int16x8_t s1_h = vmovl_s8(vget_high_s8(s1));

        float32x4_t f0_0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(s0_l))), vd);
        float32x4_t f0_1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s0_l))), vd);
        float32x4_t f0_2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(s0_h))), vd);
        float32x4_t f0_3 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s0_h))), vd);

        vst1q_f32(y + 0,  f0_0);
        vst1q_f32(y + 4,  f0_1);
        vst1q_f32(y + 8,  f0_2);
        vst1q_f32(y + 12, f0_3);

        float32x4_t f1_0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(s1_l))), vd);
        float32x4_t f1_1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s1_l))), vd);
        float32x4_t f1_2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(s1_h))), vd);
        float32x4_t f1_3 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s1_h))), vd);

        vst1q_f32(y + 16, f1_0);
        vst1q_f32(y + 20, f1_1);
        vst1q_f32(y + 24, f1_2);
        vst1q_f32(y + 28, f1_3);

        y += 32;
    }
}
#endif

void dequantize_q4_0(const void* src, float* dst, size_t n_elements) {
    const size_t nb = n_elements / QK4_0;
    const auto* blocks = static_cast<const block_q4_0*>(src);

#if defined(__ARM_NEON)
    dequantize_q4_0_neon(blocks, dst, nb);
#else
    for (size_t i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(blocks[i].d);
        float* y = dst + i * 32;
        for (size_t l = 0; l < 16; ++l) {
            const uint8_t q = blocks[i].qs[l];
            y[l + 0]  = static_cast<float>(static_cast<int>(q & 0x0F) - 8) * d;
            y[l + 16] = static_cast<float>(static_cast<int>(q >> 4)   - 8) * d;
        }
    }
#endif
}

static inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t& sc, uint8_t& m) {
    if (j < 4) {
        sc = q[j] & 63;
        m  = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        m  = (q[j + 4] >> 4)   | ((q[j - 0] >> 6) << 4);
    }
}

void dequantize_q4_K(const void* src, float* dst, size_t n_elements) {
    const size_t nb = n_elements / QK_K;
    const auto* blocks = static_cast<const block_q4_K*>(src);

    for (size_t i = 0; i < nb; ++i) {
        const auto& b = blocks[i];
        const float d = fp16_to_fp32(b.d);
        const float dmin = fp16_to_fp32(b.dmin);

        const uint8_t* qs = b.qs;
        float* y = dst + i * 256;

        for (int j = 0; j < 8; ++j) {
            uint8_t sc, m;
            get_scale_min_k4(j, b.scales, sc, m);
            const float dl = d * static_cast<float>(sc);
            const float ml = dmin * static_cast<float>(m);

            const uint8_t* sub_q = qs + j * 16;
            float* sub_y = y + j * 32;

            for (int l = 0; l < 16; ++l) {
                const uint8_t qval = sub_q[l];
                sub_y[l + 0]  = dl * static_cast<float>(qval & 0x0F) - ml;
                sub_y[l + 16] = dl * static_cast<float>(qval >> 4)   - ml;
            }
        }
    }
}

void dequantize_tensor(GgmlType type, const void* src, float* dst, size_t n_elements) {
    switch (type) {
        case GgmlType::Q4_0:
            dequantize_q4_0(src, dst, n_elements);
            break;
        case GgmlType::Q4_K:
            dequantize_q4_K(src, dst, n_elements);
            break;
        case GgmlType::F16: {
            const auto* h = static_cast<const uint16_t*>(src);
            for (size_t i = 0; i < n_elements; ++i) {
                dst[i] = fp16_to_fp32(h[i]);
            }
            break;
        }
        case GgmlType::F32:
            std::memcpy(dst, src, n_elements * sizeof(float));
            break;
        default:
            // Fallback: decode as Q4_0
            dequantize_q4_0(src, dst, n_elements);
            break;
    }
}

void quantize_row_q4_0(const float* src, void* dst, size_t n_elements) {
    const size_t nb = n_elements / QK4_0;
    auto* blocks = static_cast<block_q4_0*>(dst);

    for (size_t i = 0; i < nb; ++i) {
        const float* x = src + i * 32;
        float max_val = 0.0f;
        for (size_t l = 0; l < 32; ++l) {
            float a = std::fabs(x[l]);
            if (a > max_val) max_val = a;
        }

        const float d = max_val / -8.0f != 0.0f ? max_val / 7.0f : 1.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        blocks[i].d = fp32_to_fp16(d);

        for (size_t l = 0; l < 16; ++l) {
            const float x0 = x[l + 0] * id;
            const float x1 = x[l + 16] * id;

            int q0 = std::min(15, std::max(0, static_cast<int>(std::round(x0)) + 8));
            int q1 = std::min(15, std::max(0, static_cast<int>(std::round(x1)) + 8));

            blocks[i].qs[l] = static_cast<uint8_t>(q0 | (q1 << 4));
        }
    }
}

void quantize_row_q4_K(const float* src, void* dst, size_t n_elements) {
    const size_t nb = n_elements / QK_K;
    auto* blocks = static_cast<block_q4_K*>(dst);

    for (size_t i = 0; i < nb; ++i) {
        const float* x = src + i * 256;
        float min_val = x[0], max_val = x[0];
        for (size_t l = 1; l < 256; ++l) {
            if (x[l] < min_val) min_val = x[l];
            if (x[l] > max_val) max_val = x[l];
        }

        float d = (max_val - min_val) / 63.0f;
        if (d < 1e-6f) d = 1e-6f;
        blocks[i].d = fp32_to_fp16(d);
        blocks[i].dmin = fp32_to_fp16(min_val);

        // Pack dummy scales and quantize 8 sub-blocks of 32
        for (int j = 0; j < 8; ++j) {
            if (j < 4) {
                blocks[i].scales[j] = 16;      // scale
                blocks[i].scales[j + 4] = 0;   // min
            } else {
                blocks[i].scales[j + 4] = 16;
            }

            const float* sub_x = x + j * 32;
            uint8_t* sub_q = blocks[i].qs + j * 16;

            for (int l = 0; l < 16; ++l) {
                int q0 = std::min(15, std::max(0, static_cast<int>((sub_x[l + 0] - min_val) / (d * 16.0f) * 15.0f)));
                int q1 = std::min(15, std::max(0, static_cast<int>((sub_x[l + 16] - min_val) / (d * 16.0f) * 15.0f)));
                sub_q[l] = static_cast<uint8_t>(q0 | (q1 << 4));
            }
        }
    }
}

} // namespace streamvex
