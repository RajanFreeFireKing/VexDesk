#include "gemv.hpp"

#include <vector>
#include <thread>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <iostream>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace streamvex {

namespace {

int get_effective_threads(int requested) {
    if (requested > 0) return requested;
    unsigned int hardware = std::thread::hardware_concurrency();
    return hardware > 0 ? static_cast<int>(hardware) : 4;
}

inline float dot_product_f32(const float* a, const float* b, int n) {
#if defined(__ARM_NEON)
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);

    int i = 0;
    for (; i <= n - 16; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i + 0);
        float32x4_t b0 = vld1q_f32(b + i + 0);
        sum0 = vmlaq_f32(sum0, a0, b0);

        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        sum1 = vmlaq_f32(sum1, a1, b1);

        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        sum2 = vmlaq_f32(sum2, a2, b2);

        float32x4_t a3 = vld1q_f32(a + i + 12);
        float32x4_t b3 = vld1q_f32(b + i + 12);
        sum3 = vmlaq_f32(sum3, a3, b3);
    }
    sum0 = vaddq_f32(sum0, sum1);
    sum2 = vaddq_f32(sum2, sum3);
    sum0 = vaddq_f32(sum0, sum2);

    float total = vaddvq_f32(sum0);
    for (; i < n; ++i) {
        total += a[i] * b[i];
    }
    return total;
#else
    float total = 0.0f;
    for (int i = 0; i < n; ++i) {
        total += a[i] * b[i];
    }
    return total;
#endif
}

inline float dot_product_q4_0_block(const block_q4_0& b, const float* x) {
    const float d = fp16_to_fp32(b.d);

#if defined(__ARM_NEON)
    uint8x16_t q = vld1q_u8(b.qs);
    uint8x16_t q0 = vandq_u8(q, vdupq_n_u8(0x0F));
    uint8x16_t q1 = vshrq_n_u8(q, 4);

    int8x16_t s0 = vsubq_s8(vreinterpretq_s8_u8(q0), vdupq_n_s8(8));
    int8x16_t s1 = vsubq_s8(vreinterpretq_s8_u8(q1), vdupq_n_s8(8));

    int16x8_t s0_l = vmovl_s8(vget_low_s8(s0));
    int16x8_t s0_h = vmovl_s8(vget_high_s8(s0));
    int16x8_t s1_l = vmovl_s8(vget_low_s8(s1));
    int16x8_t s1_h = vmovl_s8(vget_high_s8(s1));

    float32x4_t f0_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s0_l)));
    float32x4_t f0_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s0_l)));
    float32x4_t f0_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s0_h)));
    float32x4_t f0_3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s0_h)));

    float32x4_t f1_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s1_l)));
    float32x4_t f1_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s1_l)));
    float32x4_t f1_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s1_h)));
    float32x4_t f1_3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s1_h)));

    float32x4_t x0_0 = vld1q_f32(x + 0);
    float32x4_t x0_1 = vld1q_f32(x + 4);
    float32x4_t x0_2 = vld1q_f32(x + 8);
    float32x4_t x0_3 = vld1q_f32(x + 12);

    float32x4_t x1_0 = vld1q_f32(x + 16);
    float32x4_t x1_1 = vld1q_f32(x + 20);
    float32x4_t x1_2 = vld1q_f32(x + 24);
    float32x4_t x1_3 = vld1q_f32(x + 28);

    float32x4_t acc0 = vmulq_f32(f0_0, x0_0);
    acc0 = vmlaq_f32(acc0, f0_1, x0_1);
    acc0 = vmlaq_f32(acc0, f0_2, x0_2);
    acc0 = vmlaq_f32(acc0, f0_3, x0_3);

    float32x4_t acc1 = vmulq_f32(f1_0, x1_0);
    acc1 = vmlaq_f32(acc1, f1_1, x1_1);
    acc1 = vmlaq_f32(acc1, f1_2, x1_2);
    acc1 = vmlaq_f32(acc1, f1_3, x1_3);

    float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
    return sum * d;
#else
    float sum = 0.0f;
    for (int l = 0; l < 16; ++l) {
        uint8_t qval = b.qs[l];
        int w0 = static_cast<int>(qval & 0x0F) - 8;
        int w1 = static_cast<int>(qval >> 4)   - 8;
        sum += (w0 * x[l + 0] + w1 * x[l + 16]);
    }
    return sum * d;
#endif
}

} // anonymous namespace

void gemv_f32(
    const float* A,
    const float* x,
    float* y,
    int rows,
    int cols,
    int num_threads
) {
    int nthreads = get_effective_threads(num_threads);
    if (nthreads <= 1 || rows < 32) {
        for (int r = 0; r < rows; ++r) {
            y[r] = dot_product_f32(A + r * cols, x, cols);
        }
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(nthreads);

    int rows_per_thread = (rows + nthreads - 1) / nthreads;
    for (int t = 0; t < nthreads; ++t) {
        int start_r = t * rows_per_thread;
        int end_r = std::min(rows, start_r + rows_per_thread);
        if (start_r >= end_r) break;

        workers.emplace_back([=]() {
            for (int r = start_r; r < end_r; ++r) {
                y[r] = dot_product_f32(A + r * cols, x, cols);
            }
        });
    }

    for (auto& w : workers) {
        if (w.joinable()) w.join();
    }
}

void gemv_q4_0_f32(
    const void* quantized_A,
    const float* x,
    float* y,
    int rows,
    int cols,
    int num_threads
) {
    const auto* blocks_A = static_cast<const block_q4_0*>(quantized_A);
    const int nb = cols / 32;

    int nthreads = get_effective_threads(num_threads);
    if (nthreads <= 1 || rows < 32) {
        for (int r = 0; r < rows; ++r) {
            float row_sum = 0.0f;
            const block_q4_0* row_blocks = blocks_A + r * nb;
            for (int b = 0; b < nb; ++b) {
                row_sum += dot_product_q4_0_block(row_blocks[b], x + b * 32);
            }
            y[r] = row_sum;
        }
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(nthreads);

    int rows_per_thread = (rows + nthreads - 1) / nthreads;
    for (int t = 0; t < nthreads; ++t) {
        int start_r = t * rows_per_thread;
        int end_r = std::min(rows, start_r + rows_per_thread);
        if (start_r >= end_r) break;

        workers.emplace_back([=]() {
            for (int r = start_r; r < end_r; ++r) {
                float row_sum = 0.0f;
                const block_q4_0* row_blocks = blocks_A + r * nb;
                for (int b = 0; b < nb; ++b) {
                    row_sum += dot_product_q4_0_block(row_blocks[b], x + b * 32);
                }
                y[r] = row_sum;
            }
        });
    }

    for (auto& w : workers) {
        if (w.joinable()) w.join();
    }
}

GemvBenchmarkResult benchmark_gemv(int rows, int cols, int iterations) {
    GemvBenchmarkResult res;
    res.rows = rows;
    res.cols = cols;
    res.num_threads = get_effective_threads(0);

    // Prepare test vectors
    std::vector<float> x(cols, 1.0f);
    std::vector<float> y(rows, 0.0f);
    std::vector<float> A(rows * cols);
    for (size_t i = 0; i < A.size(); ++i) {
        A[i] = std::sin(static_cast<float>(i) * 0.01f) * 0.1f;
    }

    // 1. Benchmark GEMV Float32
    // Warmup
    gemv_f32(A.data(), x.data(), y.data(), rows, cols, res.num_threads);

    auto t0_f32 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iterations; ++it) {
        gemv_f32(A.data(), x.data(), y.data(), rows, cols, res.num_threads);
    }
    auto t1_f32 = std::chrono::high_resolution_clock::now();
    double total_f32_sec = std::chrono::duration<double>(t1_f32 - t0_f32).count();
    res.time_f32_ms = (total_f32_sec / iterations) * 1000.0;

    double flops = 2.0 * static_cast<double>(rows) * static_cast<double>(cols);
    res.gflops_f32 = (flops / 1e9) / (res.time_f32_ms / 1000.0);

    double bytes_f32 = static_cast<double>(rows * cols) * sizeof(float) + static_cast<double>(cols + rows) * sizeof(float);
    res.bandwidth_f32_gb_s = (bytes_f32 / (1024.0 * 1024.0 * 1024.0)) / (res.time_f32_ms / 1000.0);

    // 2. Benchmark Direct Quantized GEMV Q4_0
    size_t nb_total = (rows * cols) / 32;
    std::vector<block_q4_0> qA(nb_total);
    quantize_row_q4_0(A.data(), qA.data(), rows * cols);

    // Warmup
    gemv_q4_0_f32(qA.data(), x.data(), y.data(), rows, cols, res.num_threads);

    auto t0_q4 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iterations; ++it) {
        gemv_q4_0_f32(qA.data(), x.data(), y.data(), rows, cols, res.num_threads);
    }
    auto t1_q4 = std::chrono::high_resolution_clock::now();
    double total_q4_sec = std::chrono::duration<double>(t1_q4 - t0_q4).count();
    res.time_q4_0_ms = (total_q4_sec / iterations) * 1000.0;
    res.gflops_q4_0 = (flops / 1e9) / (res.time_q4_0_ms / 1000.0);

    double bytes_q4 = static_cast<double>(nb_total) * sizeof(block_q4_0) + static_cast<double>(cols + rows) * sizeof(float);
    res.bandwidth_q4_0_gb_s = (bytes_q4 / (1024.0 * 1024.0 * 1024.0)) / (res.time_q4_0_ms / 1000.0);

    return res;
}

} // namespace streamvex
