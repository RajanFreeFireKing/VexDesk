#ifndef STREAMVEX_H
#define STREAMVEX_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
    #ifdef STREAMVEX_BUILDING_DLL
        #define STREAMVEX_API __declspec(dllexport)
    #else
        #define STREAMVEX_API __declspec(dllimport)
    #endif
#else
    #if __GNUC__ >= 4
        #define STREAMVEX_API __attribute__((visibility("default")))
    #else
        #define STREAMVEX_API
    #endif
#endif

/**
 * Token callback function type.
 * Invoked synchronously or from worker thread for each generated token string.
 */
typedef void (*TokenCallback)(const char* token);

/**
 * Initializes the local LLM engine with the given model path (e.g. .gguf file).
 * 
 * @param model_path Null-terminated string pointing to local model weights.
 * @return 0 on success, non-zero error code if initialization fails.
 */
STREAMVEX_API int init_engine(const char* model_path);

/**
 * Streams tokens for the given prompt to the provided callback function.
 * 
 * @param prompt Null-terminated user prompt / conversation string.
 * @param callback Function pointer called per generated token chunk.
 * @return 0 on completion, non-zero if generation encounters an error.
 */
STREAMVEX_API int stream_token(const char* prompt, TokenCallback callback);

/**
 * Retrieves the current resident set / RAM memory usage of the engine in Megabytes (MB).
 * 
 * @return RAM usage in MB.
 */
STREAMVEX_API double get_memory_usage(void);

/**
 * Checks whether the engine has been successfully loaded with a model.
 * 
 * @return 1 if ready, 0 otherwise.
 */
STREAMVEX_API int is_engine_ready(void);

/**
 * Releases model weights and frees allocated engine memory.
 */
STREAMVEX_API void release_engine(void);

/**
 * Retrieves the latest generation telemetry summary (TTFT, tokens/sec, peak RAM) in JSON format.
 * 
 * @param out_json Buffer to receive JSON string.
 * @param max_len Maximum capacity of out_json buffer.
 * @return 0 on success, non-zero on error.
 */
STREAMVEX_API int get_last_generation_telemetry(char* out_json, size_t max_len);

/**
 * Runs the asynchronous double-buffering layer streaming benchmark.
 * 
 * @param file_path Path to model file, or NULL / empty for synthetic benchmarking.
 * @param chunk_size_mb Size of each layer chunk in Megabytes (e.g. 50 or 64).
 * @param total_layers Total number of layers to stream sequentially.
 * @param out_json_stats Buffer to receive formatted JSON performance metrics.
 * @param max_len Maximum length of out_json_stats buffer.
 * @return 0 on success, non-zero on failure.
 */
STREAMVEX_API int run_streaming_benchmark(
    const char* file_path,
    size_t chunk_size_mb,
    int total_layers,
    char* out_json_stats,
    size_t max_len
);

/**
 * Parses GGUF model metadata and tensor catalog from disk without loading tensor data.
 * Outputs formatted JSON into out_json_info.
 * 
 * @param model_path Path to the local .gguf file.
 * @param out_json_info Output buffer for JSON summary.
 * @param max_len Maximum length of out_json_info buffer.
 * @return 0 on success, non-zero on failure.
 */
STREAMVEX_API int parse_gguf_metadata(
    const char* model_path,
    char* out_json_info,
    size_t max_len
);

/**
 * Streams a specific layer tensor from disk and dequantizes it to float32
 * using optimized SIMD/NEON kernels.
 * 
 * @param model_path Path to the local .gguf file.
 * @param tensor_name Name of the target tensor (e.g. "blk.0.attn_q.weight").
 * @param out_buffer Output float array to receive dequantized weights.
 * @param max_floats Maximum number of float elements out_buffer can store.
 * @return Number of dequantized floats on success, negative error code on failure.
 */
STREAMVEX_API int stream_and_dequant_layer(
    const char* model_path,
    const char* tensor_name,
    float* out_buffer,
    size_t max_floats
);

/**
 * Runs high-performance Matrix-Vector Multiplication (GEMV) benchmark comparing
 * single-precision float32 vs direct 4-bit quantized (Q4_0) execution.
 * 
 * @param rows Number of matrix rows (e.g. 2048 or 4096).
 * @param cols Number of matrix columns (e.g. 2048 or 4096).
 * @param out_json_stats Output buffer for JSON summary containing GFLOPS and bandwidth.
 * @param max_len Maximum length of out_json_stats buffer.
 * @return 0 on success, non-zero on failure.
 */
STREAMVEX_API int compute_gemv_benchmark(
    int rows,
    int cols,
    char* out_json_stats,
    size_t max_len
);

/**
 * Executes a single autoregressive Transformer forward pass step:
 * Embedding lookup, layer attention with KV-cache, SwiGLU FFN, RMSNorm,
 * logits projection, and Top-K/Top-P token sampling.
 * 
 * @param token_id Current input token ID.
 * @param pos Current sequence position index (0, 1, 2, ...).
 * @param out_next_token Output pointer to receive predicted next token ID.
 * @param out_ms Output pointer to receive forward pass step execution latency in milliseconds.
 * @return 0 on success, non-zero on error.
 */
STREAMVEX_API int run_forward_pass_step(
    int token_id,
    int pos,
    int* out_next_token,
    float* out_ms
);

#ifdef __cplusplus
}
#endif

#endif // STREAMVEX_H
