#ifndef GEMV_HPP
#define GEMV_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include "dequant.hpp"

namespace streamvex {

struct GemvBenchmarkResult {
    int rows = 0;
    int cols = 0;
    double time_f32_ms = 0.0;
    double gflops_f32 = 0.0;
    double bandwidth_f32_gb_s = 0.0;
    double time_q4_0_ms = 0.0;
    double gflops_q4_0 = 0.0;
    double bandwidth_q4_0_gb_s = 0.0;
    int num_threads = 1;
};

/**
 * High-performance Float32 Matrix-Vector Multiplication: y = A * x
 * 
 * @param A Pointer to row-major rows x cols matrix.
 * @param x Input vector of length cols.
 * @param y Output vector of length rows.
 * @param rows Number of rows in matrix A.
 * @param cols Number of columns in matrix A (must be a multiple of 4).
 * @param num_threads Number of worker threads (0 = auto-detect CPU cores).
 */
void gemv_f32(
    const float* A,
    const float* x,
    float* y,
    int rows,
    int cols,
    int num_threads = 0
);

/**
 * Direct Quantized 4-Bit (Q4_0) Matrix-Vector Multiplication: y = Quantized_A * x
 * Multiplies compressed 4-bit weights directly with float activations to reduce DRAM traffic by 7x.
 * 
 * @param quantized_A Pointer to row-major block_q4_0 rows x (cols / 32) blocks.
 * @param x Input vector of length cols.
 * @param y Output vector of length rows.
 * @param rows Number of rows in matrix A.
 * @param cols Number of columns in matrix A (must be a multiple of 32).
 * @param num_threads Number of worker threads (0 = auto-detect CPU cores).
 */
void gemv_q4_0_f32(
    const void* quantized_A,
    const float* x,
    float* y,
    int rows,
    int cols,
    int num_threads = 0
);

/**
 * Benchmarks both f32 and q4_0 GEMV kernels on the current hardware.
 */
GemvBenchmarkResult benchmark_gemv(int rows = 2048, int cols = 2048, int iterations = 5);

} // namespace streamvex

#endif // GEMV_HPP
