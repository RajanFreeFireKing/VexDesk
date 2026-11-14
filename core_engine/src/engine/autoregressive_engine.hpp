#ifndef AUTOREGRESSIVE_ENGINE_HPP
#define AUTOREGRESSIVE_ENGINE_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>
#include <mutex>
#include <chrono>

#include "../io/layer_streamer.hpp"
#include "../parser/gguf_reader.hpp"
#include "../compute/dequant.hpp"
#include "../compute/gemv.hpp"
#include "../compute/transformer_layer.hpp"
#include "../compute/sampler.hpp"

namespace streamvex {

/**
 * Embedded BPE / Byte-Fallback Tokenizer & UTF-8 Decoder.
 * Maps prompt strings to token sequences, and decodes autoregressively generated
 * token IDs back into UTF-8 text strings with byte-level fallback.
 */
class Tokenizer {
public:
    static constexpr int TOKEN_UNK = 0;
    static constexpr int TOKEN_BOS = 1;
    static constexpr int TOKEN_EOS = 2;
    static constexpr int TOKEN_PAD = 3;
    static constexpr int TOKEN_IM_START = 4;
    static constexpr int TOKEN_IM_END = 5;
    static constexpr int BYTE_OFFSET = 6; // Tokens 6..261 are raw bytes 0x00..0xFF

    Tokenizer();

    std::vector<int> encode(const std::string& text) const;
    std::string decode_token(int token_id) const;
    std::string decode_tokens(const std::vector<int>& token_ids) const;

    int vocab_size() const { return static_cast<int>(m_id_to_token.size()); }
    int find_token(const std::string& token_str) const;

private:
    void init_default_vocabulary();
    void add_token(int id, const std::string& token_str);

    std::vector<std::string> m_id_to_token;
    std::unordered_map<std::string, int> m_token_to_id;
};

/**
 * Live inference and telemetry statistics collected during autoregressive generation.
 */
struct GenerationTelemetry {
    int total_tokens = 0;
    float time_to_first_token_ms = 0.0f;
    float total_duration_sec = 0.0f;
    float tokens_per_sec = 0.0f;
    double baseline_ram_mb = 0.0f;
    double peak_ram_mb = 0.0f;
    double current_ram_mb = 0.0f;
    int layers_streamed = 0;
    double disk_throughput_mb_s = 0.0f;
    bool is_ram_bounded = true;
    std::string model_name;
};

/**
 * Unified Autoregressive Engine Pipeline for StreamVex.
 * 
 * Integrates:
 * - GgufReader: Reads GGUF metadata, tensor dictionaries, and weight offsets.
 * - LayerStreamer: Asynchronous double-buffered layer prefetching keeping resident RAM capped.
 * - GEMV & TransformerLayer: Vectorized ARM NEON / SIMD forward pass with compact KV-cache.
 * - Sampler: Temperature, Top-K, and Top-P token sampling.
 * - Tokenizer: Embedded BPE / byte-fallback UTF-8 encoding and decoding.
 */
class AutoregressiveEngine {
public:
    AutoregressiveEngine();
    ~AutoregressiveEngine();

    // Prevent copies
    AutoregressiveEngine(const AutoregressiveEngine&) = delete;
    AutoregressiveEngine& operator=(const AutoregressiveEngine&) = delete;

    /**
     * Initializes the engine with the specified model file path (.gguf or simulated).
     * Opens GGUF, configures layer streaming parameters, pre-allocates compact KV-cache,
     * and sets up the embedding / vocab matrices.
     * 
     * @param model_path Path to local model file.
     * @return true on success, false on error.
     */
    bool initialize(const std::string& model_path);

    /**
     * Executes the complete autoregressive generation pipeline:
     * 1. Tokenizes user prompt via embedded BPE / byte fallback.
     * 2. Prefill phase: feeds prompt tokens through Transformer forward pass.
     * 3. Decode loop: step-by-step token generation.
     *    In each step:
     *    - Streams layer weights using asynchronous double-buffer (RAM capped to 2 buffers).
     *    - Computes RMSNorm, RoPE, Attention (KV-cache), and SwiGLU FFN.
     *    - Samples next token and decodes it to UTF-8 text chunk.
     *    - Invokes callback with decoded token piece.
     * 
     * @param prompt Null-terminated user prompt.
     * @param callback Function called per generated UTF-8 token piece.
     * @param max_tokens Maximum number of tokens to generate (default 96).
     * @return 0 on success, non-zero error code otherwise.
     */
    int generate_stream(
        const std::string& prompt,
        std::function<void(const char* token)> callback,
        int max_tokens = 256
    );

    /**
     * Queries current resident memory usage in Megabytes (MB).
     */
    double get_memory_usage() const;

    /**
     * Checks if engine has been loaded with a model.
     */
    bool is_ready() const;

    /**
     * Releases loaded model weights, streamer buffers, and KV-cache.
     */
    void release();

    /**
     * Retrieves the latest generation telemetry metrics.
     */
    GenerationTelemetry get_telemetry() const;

    /**
     * Formats current/last telemetry stats as a JSON string.
     */
    std::string get_telemetry_json() const;

    const std::string& model_path() const { return m_model_path; }

private:
    void init_synthetic_weights();
    std::string construct_response_plan(const std::string& prompt);

    mutable std::mutex m_engine_mtx;
    bool m_is_initialized = false;
    std::string m_model_path;

    TransformerConfig m_config;
    std::unique_ptr<GgufReader> m_gguf_reader;
    std::unique_ptr<LayerStreamer> m_streamer;
    std::unique_ptr<CompactKVCache> m_kv_cache;

    LayerWeights m_layer_weights;
    std::vector<block_q4_0> m_vocab_weights_q4;
    std::vector<float> m_final_norm_weights;
    std::vector<float> m_embedding_table;

    Sampler m_sampler;
    Tokenizer m_tokenizer;
    GenerationTelemetry m_telemetry;

    size_t m_layer_chunk_size_bytes = 0;
    std::string m_synthetic_file_path;
};

} // namespace streamvex

#endif // AUTOREGRESSIVE_ENGINE_HPP
