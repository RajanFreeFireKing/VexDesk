#ifndef DEQUANT_HPP
#define DEQUANT_HPP

#include <cstdint>
#include <cstddef>
#include <cstring>
#include "../parser/gguf_reader.hpp"

namespace streamvex {

constexpr size_t QK4_0 = 32;
constexpr size_t QK_K  = 256;

// Binary layout of a Q4_0 block (18 bytes per 32 elements)
struct block_q4_0 {
    uint16_t d;        // Half-precision float delta
    uint8_t qs[16];    // 32 4-bit nibbles (low nibble first)
};

// Binary layout of a Q4_K block (144 bytes per 256 elements)
struct block_q4_K {
    uint16_t d;          // Super-block scale (fp16)
    uint16_t dmin;       // Super-block min (fp16)
    uint8_t scales[12];  // 6-bit scales & mins for 8 sub-blocks
    uint8_t qs[128];     // 256 4-bit quants
};

/**
 * Fast IEEE-754 half-precision float to single-precision float conversion.
 */
static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000) << 16;
    int32_t exp   = (h & 0x7C00) >> 10;
    uint32_t mant = (h & 0x03FF);

    if (exp == 0) {
        if (mant == 0) {
            uint32_t val = sign;
            float f;
            std::memcpy(&f, &val, sizeof(f));
            return f;
        }
        while (!(mant & 0x0400)) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= ~0x0400;
    } else if (exp == 31) {
        exp = 255;
    } else {
        exp = exp - 15 + 127;
    }

    uint32_t val = sign | (static_cast<uint32_t>(exp) << 23) | (mant << 13);
    float f;
    std::memcpy(&f, &val, sizeof(f));
    return f;
}

/**
 * Single-precision float to half-precision float conversion.
 */
static inline uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (x & 0x007FFFFF);

    if (exp <= 0) {
        return static_cast<uint16_t>(sign);
    } else if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

/**
 * Dequantizes a stream of Q4_0 blocks to float32.
 * Highly optimized with ARM NEON SIMD vectorization when running on aarch64.
 * 
 * @param src Pointer to contiguous block_q4_0 memory chunks.
 * @param dst Pointer to float output array.
 * @param n_elements Number of weights to dequantize (must be a multiple of 32).
 */
void dequantize_q4_0(const void* src, float* dst, size_t n_elements);

/**
 * Dequantizes a stream of Q4_K super-blocks to float32.
 * 
 * @param src Pointer to contiguous block_q4_K memory chunks.
 * @param dst Pointer to float output array.
 * @param n_elements Number of weights to dequantize (must be a multiple of 256).
 */
void dequantize_q4_K(const void* src, float* dst, size_t n_elements);

/**
 * Generalized tensor dequantization dispatcher.
 */
void dequantize_tensor(GgmlType type, const void* src, float* dst, size_t n_elements);

/**
 * Reference quantization routines to synthesize valid Q4_0 and Q4_K blocks for testing.
 */
void quantize_row_q4_0(const float* src, void* dst, size_t n_elements);
void quantize_row_q4_K(const float* src, void* dst, size_t n_elements);

} // namespace streamvex

#endif // DEQUANT_HPP
