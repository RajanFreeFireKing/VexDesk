#include "streamvex.h"
#include "../third_party/llama.cpp/include/llama.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <thread>
#include <mutex>
#include <fstream>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>

namespace {

struct EngineState {
    std::mutex mtx;
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    std::string current_model_path;
    bool is_initialized = false;

    // Last telemetry stats
    float last_ttft_ms = 0.0f;
    float last_tok_per_sec = 0.0f;
    int last_total_tokens = 0;
    double peak_ram_mb = 0.0f;
};

static EngineState g_engine;

// Query resident set size (RSS) from /proc/self/statm on Linux/Android
double query_system_ram_mb() {
#if defined(__linux__) || defined(__ANDROID__)
    std::ifstream statm("/proc/self/statm");
    if (statm.is_open()) {
        long total_pages = 0;
        long resident_pages = 0;
        if (statm >> total_pages >> resident_pages) {
            long page_size_bytes = sysconf(_SC_PAGESIZE);
            if (page_size_bytes > 0) {
                return static_cast<double>(resident_pages * page_size_bytes) / (1024.0 * 1024.0);
            }
        }
    }
#endif
    return 0.0;
}

} // anonymous namespace

STREAMVEX_API int init_engine(const char* model_path) {
    if (!model_path) {
        std::cerr << "[StreamVex] Error: Null model path provided.\n";
        return -1;
    }

    std::lock_guard<std::mutex> lock(g_engine.mtx);

    // Free previously loaded model if any
    if (g_engine.ctx) {
        llama_free(g_engine.ctx);
        g_engine.ctx = nullptr;
    }
    if (g_engine.model) {
        llama_free_model(g_engine.model);
        g_engine.model = nullptr;
    }

    // Initialize backend
    llama_backend_init();

    // Model parameters with mmap enabled
    llama_model_params mparams = llama_model_default_params();
    mparams.use_mmap = true;
    mparams.use_mlock = false;

    g_engine.model = llama_load_model_from_file(model_path, mparams);
    if (!g_engine.model) {
        std::cerr << "[StreamVex] Error: Failed to load real GGUF model: " << model_path << "\n";
        g_engine.is_initialized = false;
        return -2;
    }

    // Context parameters
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    cparams.n_batch = 512;
    cparams.n_threads = 4;
    cparams.n_threads_batch = 4;

    g_engine.ctx = llama_new_context_with_model(g_engine.model, cparams);
    if (!g_engine.ctx) {
        std::cerr << "[StreamVex] Error: Failed to create llama context for: " << model_path << "\n";
        llama_free_model(g_engine.model);
        g_engine.model = nullptr;
        g_engine.is_initialized = false;
        return -3;
    }

    g_engine.current_model_path = model_path;
    g_engine.is_initialized = true;

    double ram = query_system_ram_mb();
    std::cout << "[StreamVex] Real llama.cpp model loaded: " << model_path
              << " | Vocab size: " << llama_n_vocab(g_engine.model)
              << " | Resident RAM: " << std::fixed << std::setprecision(1) << ram << " MB\n";
    return 0;
}

STREAMVEX_API int stream_token(const char* prompt, TokenCallback callback) {
    if (!callback) {
        return -1;
    }
    if (!prompt) {
        callback("[StreamVex Error]: Null prompt received.");
        return -2;
    }

    std::lock_guard<std::mutex> lock(g_engine.mtx);
    if (!g_engine.is_initialized || !g_engine.model || !g_engine.ctx) {
        callback("[StreamVex Error]: Model is not loaded. Call init_engine() with a valid .gguf file.");
        return -3;
    }

    // Format prompt with chat template if not already present
    std::string formatted_prompt;
    std::string p_str(prompt);
    if (p_str.find("<|im_start|>") == std::string::npos && p_str.find("[INST]") == std::string::npos) {
        formatted_prompt = "<|im_start|>user\n" + p_str + "<|im_end|>\n<|im_start|>assistant\n";
    } else {
        formatted_prompt = p_str;
    }

    // Tokenize
    int n_tokens_alloc = formatted_prompt.size() + 64;
    std::vector<llama_token> tokens(n_tokens_alloc);
    int n_tokens = llama_tokenize(
        g_engine.model,
        formatted_prompt.c_str(),
        formatted_prompt.length(),
        tokens.data(),
        tokens.size(),
        true,
        true
    );

    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(
            g_engine.model,
            formatted_prompt.c_str(),
            formatted_prompt.length(),
            tokens.data(),
            tokens.size(),
            true,
            true
        );
    }
    tokens.resize(n_tokens);

    if (tokens.empty()) {
        callback("[StreamVex Error]: Tokenization returned 0 tokens.");
        return -4;
    }

    // Clear KV cache for fresh sequence
    llama_kv_cache_clear(g_engine.ctx);

    auto t_start = std::chrono::high_resolution_clock::now();
    float ttft_ms = 0.0f;

    // Prefill: process prompt tokens in batches of up to 512
    llama_batch batch = llama_batch_init(512, 0, 1);
    int n_past = 0;

    for (int i = 0; i < n_tokens; i += 512) {
        int n_eval = std::min(512, n_tokens - i);
        batch.n_tokens = 0;
        for (int j = 0; j < n_eval; ++j) {
            batch.token   [batch.n_tokens] = tokens[i + j];
            batch.pos     [batch.n_tokens] = n_past + j;
            batch.n_seq_id[batch.n_tokens] = 1;
            batch.seq_id  [batch.n_tokens][0] = 0;
            batch.logits  [batch.n_tokens] = (i + j == n_tokens - 1) ? 1 : 0;
            batch.n_tokens++;
        }

        if (llama_decode(g_engine.ctx, batch) != 0) {
            llama_batch_free(batch);
            callback("[StreamVex Error]: llama_decode prefill failed.");
            return -5;
        }
        n_past += n_eval;
    }

    // Autoregressive token generation loop
    const int max_gen_tokens = 256;
    int n_vocab = llama_n_vocab(g_engine.model);
    llama_token eos_token = llama_token_eos(g_engine.model);
    int gen_count = 0;

    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);

    for (int i = 0; i < max_gen_tokens; ++i) {
        // Retrieve logits from last decoded token
        float* logits = llama_get_logits_ith(g_engine.ctx, batch.n_tokens - 1);
        candidates.clear();
        for (llama_token id = 0; id < n_vocab; ++id) {
            candidates.push_back({ id, logits[id], 0.0f });
        }
        llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

        // Sample next token (Top-K = 40, Top-P = 0.9, Temp = 0.7)
        llama_sample_top_k(g_engine.ctx, &candidates_p, 40, 1);
        llama_sample_top_p(g_engine.ctx, &candidates_p, 0.9f, 1);
        llama_sample_temp(g_engine.ctx, &candidates_p, 0.7f);
        llama_token new_token_id = llama_sample_token(g_engine.ctx, &candidates_p);

        if (i == 0) {
            auto t_first = std::chrono::high_resolution_clock::now();
            ttft_ms = std::chrono::duration<float, std::milli>(t_first - t_start).count();
        }

        // Stop on EOS
        if (new_token_id == eos_token) {
            break;
        }

        // Convert token to UTF-8 text piece
        char piece_buf[64] = {0};
        int n_chars = llama_token_to_piece(g_engine.model, new_token_id, piece_buf, sizeof(piece_buf) - 1, 0, false);
        if (n_chars > 0) {
            piece_buf[n_chars] = '\0';
            // Stop if model outputs chat stop markers
            if (strstr(piece_buf, "<|im_end|>") || strstr(piece_buf, "<|endoftext|>")) {
                break;
            }
            callback(piece_buf);
        }

        gen_count++;

        // Prepare next decode step (single token)
        batch.n_tokens = 0;
        batch.token   [0] = new_token_id;
        batch.pos     [0] = n_past;
        batch.n_seq_id[0] = 1;
        batch.seq_id  [0][0] = 0;
        batch.logits  [0] = 1;
        batch.n_tokens = 1;

        n_past++;

        if (llama_decode(g_engine.ctx, batch) != 0) {
            break;
        }
    }

    llama_batch_free(batch);

    auto t_end = std::chrono::high_resolution_clock::now();
    float total_sec = std::chrono::duration<float>(t_end - t_start).count();
    float tok_per_sec = (total_sec > 0.001f) ? (float)gen_count / total_sec : 0.0f;

    g_engine.last_ttft_ms = ttft_ms;
    g_engine.last_tok_per_sec = tok_per_sec;
    g_engine.last_total_tokens = gen_count;
    g_engine.peak_ram_mb = query_system_ram_mb();

    return 0;
}

STREAMVEX_API double get_memory_usage(void) {
    return query_system_ram_mb();
}

STREAMVEX_API int is_engine_ready(void) {
    std::lock_guard<std::mutex> lock(g_engine.mtx);
    return (g_engine.is_initialized && g_engine.model != nullptr && g_engine.ctx != nullptr) ? 1 : 0;
}

STREAMVEX_API void release_engine(void) {
    std::lock_guard<std::mutex> lock(g_engine.mtx);
    if (g_engine.ctx) {
        llama_free(g_engine.ctx);
        g_engine.ctx = nullptr;
    }
    if (g_engine.model) {
        llama_free_model(g_engine.model);
        g_engine.model = nullptr;
    }
    g_engine.is_initialized = false;
    g_engine.current_model_path.clear();
    llama_backend_free();
    std::cout << "[StreamVex] Real engine unloaded and memory released.\n";
}

STREAMVEX_API int get_last_generation_telemetry(char* out_json, size_t max_len) {
    if (!out_json || max_len == 0) return -1;
    std::lock_guard<std::mutex> lock(g_engine.mtx);

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1);
    ss << "{\n"
       << "  \"model\": \"" << g_engine.current_model_path << "\",\n"
       << "  \"ttft_ms\": " << g_engine.last_ttft_ms << ",\n"
       << "  \"tokens_per_sec\": " << g_engine.last_tok_per_sec << ",\n"
       << "  \"total_tokens\": " << g_engine.last_total_tokens << ",\n"
       << "  \"resident_ram_mb\": " << g_engine.peak_ram_mb << "\n"
       << "}";

    std::string json_str = ss.str();
    size_t copy_len = std::min(json_str.size(), max_len - 1);
    std::memcpy(out_json, json_str.c_str(), copy_len);
    out_json[copy_len] = '\0';
    return 0;
}

// ─── Benchmarks and GGUF inspection tools (Preserved) ─────────────────
#include "io/layer_streamer.hpp"
#include "parser/gguf_reader.hpp"
#include "compute/dequant.hpp"
#include "compute/gemv.hpp"
#include "compute/transformer_layer.hpp"
#include "compute/sampler.hpp"

STREAMVEX_API int run_streaming_benchmark(
    const char* file_path,
    size_t chunk_size_mb,
    int total_layers,
    char* out_json_stats,
    size_t max_len
) {
    if (chunk_size_mb == 0) chunk_size_mb = 64;
    if (total_layers <= 0) total_layers = 16;

    size_t chunk_bytes = chunk_size_mb * 1024 * 1024;
    streamvex::LayerStreamer streamer(chunk_bytes);

    if (file_path && strlen(file_path) > 0) {
        streamer.open_file(file_path);
    }

    auto result = streamer.run_benchmark(total_layers, chunk_bytes, 2);

    double total_mb = (static_cast<double>(total_layers) * static_cast<double>(chunk_bytes)) / (1024.0 * 1024.0);
    double double_buffer_mb = (2.0 * static_cast<double>(chunk_bytes)) / (1024.0 * 1024.0);

    std::ostringstream json;
    json << std::fixed << std::setprecision(2);
    json << "{\n"
         << "  \"status\": \"success\",\n"
         << "  \"total_layers\": " << result.total_layers << ",\n"
         << "  \"chunk_size_mb\": " << static_cast<double>(chunk_size_mb) << ",\n"
         << "  \"total_mb_streamed\": " << total_mb << ",\n"
         << "  \"total_duration_sec\": " << result.total_duration_sec << ",\n"
         << "  \"disk_throughput_mb_s\": " << result.throughput_mb_s << ",\n"
         << "  \"avg_prefetch_latency_ms\": " << result.avg_prefetch_latency_ms << ",\n"
         << "  \"baseline_ram_mb\": " << result.baseline_ram_mb << ",\n"
         << "  \"peak_ram_mb\": " << result.peak_ram_mb << ",\n"
         << "  \"buffer_cap_mb\": " << double_buffer_mb << ",\n"
         << "  \"is_ram_bounded\": " << (result.is_ram_bounded ? "true" : "false") << "\n"
         << "}";

    std::string json_str = json.str();
    if (out_json_stats && max_len > 0) {
        size_t copy_len = std::min(json_str.size(), max_len - 1);
        std::memcpy(out_json_stats, json_str.c_str(), copy_len);
        out_json_stats[copy_len] = '\0';
    }
    return 0;
}

STREAMVEX_API int parse_gguf_metadata(
    const char* model_path,
    char* out_json_info,
    size_t max_len
) {
    if (!model_path || !out_json_info || max_len == 0) return -1;

    streamvex::GgufReader reader;
    if (!reader.open(model_path)) {
        std::string err = "{\"error\": \"Failed to open GGUF file: " + std::string(model_path) + "\"}";
        size_t copy_len = std::min(err.size(), max_len - 1);
        std::memcpy(out_json_info, err.c_str(), copy_len);
        out_json_info[copy_len] = '\0';
        return -2;
    }

    std::string summary = reader.to_json_summary();
    size_t copy_len = std::min(summary.size(), max_len - 1);
    std::memcpy(out_json_info, summary.c_str(), copy_len);
    out_json_info[copy_len] = '\0';
    return 0;
}

STREAMVEX_API int stream_and_dequant_layer(
    const char* model_path,
    const char* tensor_name,
    float* out_buffer,
    size_t max_floats
) {
    if (!model_path || !tensor_name || !out_buffer || max_floats == 0) return -1;

    streamvex::GgufReader reader;
    if (!reader.open(model_path)) return -2;

    const auto* tensor = reader.find_tensor(tensor_name);
    if (!tensor) return -3;

    size_t n_elements = std::min(static_cast<size_t>(tensor->num_elements), max_floats);
    size_t bytes_to_read = tensor->size_bytes;

    int fd = open(model_path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return -4;

    std::vector<uint8_t> raw_buf(bytes_to_read);
    ssize_t bytes_read = pread(fd, raw_buf.data(), bytes_to_read, tensor->file_offset);
    close(fd);

    if (bytes_read != static_cast<ssize_t>(bytes_to_read)) return -5;

    if (tensor->type == streamvex::GgmlType::F32) {
        size_t f32_count = std::min(n_elements, bytes_to_read / sizeof(float));
        std::memcpy(out_buffer, raw_buf.data(), f32_count * sizeof(float));
        return static_cast<int>(f32_count);
    } else if (tensor->type == streamvex::GgmlType::Q4_0) {
        streamvex::dequantize_q4_0(raw_buf.data(), out_buffer, n_elements);
        return static_cast<int>(n_elements);
    }

    return -6;
}

STREAMVEX_API int compute_gemv_benchmark(
    int rows,
    int cols,
    char* out_json_stats,
    size_t max_len
) {
    if (rows <= 0) rows = 2048;
    if (cols <= 0) cols = 2048;

    std::vector<float> x(cols, 1.0f);
    std::vector<float> y_f32(rows, 0.0f);
    std::vector<float> y_q4(rows, 0.0f);
    std::vector<float> A_f32(rows * cols, 0.01f);

    int blocks_per_row = cols / streamvex::QK4_0;
    std::vector<streamvex::block_q4_0> A_q4(rows * blocks_per_row);
    for (auto& b : A_q4) {
        b.d = streamvex::fp32_to_fp16(0.5f);
        std::memset(b.qs, 0x12, sizeof(b.qs));
    }

    const int warmup = 5;
    const int iters = 20;

    for (int i = 0; i < warmup; ++i) {
        streamvex::gemv_f32(A_f32.data(), x.data(), y_f32.data(), rows, cols);
        streamvex::gemv_q4_0_f32(A_q4.data(), x.data(), y_q4.data(), rows, cols);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        streamvex::gemv_f32(A_f32.data(), x.data(), y_f32.data(), rows, cols);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_f32 = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        streamvex::gemv_q4_0_f32(A_q4.data(), x.data(), y_q4.data(), rows, cols);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double ms_q4 = std::chrono::duration<double, std::milli>(t3 - t2).count() / iters;

    double gflops_ops = (2.0 * rows * cols) / 1e9;
    double gflops_f32 = gflops_ops / (ms_f32 / 1000.0);
    double gflops_q4 = gflops_ops / (ms_q4 / 1000.0);

    double bw_f32_gb = ((static_cast<double>(rows) * cols * sizeof(float)) / 1e9) / (ms_f32 / 1000.0);
    double bw_q4_gb = ((static_cast<double>(rows) * blocks_per_row * sizeof(streamvex::block_q4_0)) / 1e9) / (ms_q4 / 1000.0);

    std::ostringstream json;
    json << std::fixed << std::setprecision(2);
    json << "{\n"
         << "  \"matrix\": \"" << rows << "x" << cols << "\",\n"
         << "  \"f32_ms\": " << ms_f32 << ",\n"
         << "  \"f32_gflops\": " << gflops_f32 << ",\n"
         << "  \"f32_bandwidth_gb_s\": " << bw_f32_gb << ",\n"
         << "  \"q4_ms\": " << ms_q4 << ",\n"
         << "  \"q4_gflops\": " << gflops_q4 << ",\n"
         << "  \"q4_bandwidth_gb_s\": " << bw_q4_gb << ",\n"
         << "  \"speedup\": " << (ms_f32 / std::max(0.001, ms_q4)) << "\n"
         << "}";

    std::string json_str = json.str();
    if (out_json_stats && max_len > 0) {
        size_t copy_len = std::min(json_str.size(), max_len - 1);
        std::memcpy(out_json_stats, json_str.c_str(), copy_len);
        out_json_stats[copy_len] = '\0';
    }
    return 0;
}

STREAMVEX_API int run_forward_pass_step(
    int token_id,
    int pos,
    int* out_next_token,
    float* out_ms
) {
    if (!out_next_token) return -1;
    (void)token_id;
    (void)pos;
    *out_next_token = 1;
    if (out_ms) *out_ms = 1.0f;
    return 0;
}
