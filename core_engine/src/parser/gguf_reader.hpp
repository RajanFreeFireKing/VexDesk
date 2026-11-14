#ifndef GGUF_READER_HPP
#define GGUF_READER_HPP

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace streamvex {

// GGML Quantization Type definitions (aligned with llama.cpp/ggml)
enum class GgmlType : uint32_t {
    F32  = 0,
    F16  = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    UNKNOWN = 999
};

const char* ggml_type_name(GgmlType type);
size_t ggml_type_block_size(GgmlType type);
size_t ggml_type_type_size(GgmlType type);

// GGUF Metadata Value Types
enum class GgufValueType : uint32_t {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,
};

struct GgufTensorInfo {
    std::string name;
    uint32_t n_dimensions = 0;
    std::vector<uint64_t> dimensions;
    GgmlType type = GgmlType::UNKNOWN;
    uint64_t tensor_offset = 0;   // Offset relative to tensor data section
    uint64_t file_offset = 0;     // Absolute file offset to tensor data bytes
    uint64_t num_elements = 0;
    uint64_t size_bytes = 0;
};

class GgufReader {
public:
    GgufReader();
    ~GgufReader();

    // Prevent copy
    GgufReader(const GgufReader&) = delete;
    GgufReader& operator=(const GgufReader&) = delete;

    /**
     * Parses the GGUF file header, metadata key-value pairs, and tensor dictionary.
     * Executes zero-copy sequential parsing without loading tensor weight data into RAM.
     * 
     * @param file_path Path to the .gguf file.
     * @return true if successfully parsed and valid GGUF magic found, false otherwise.
     */
    bool open(const std::string& file_path);

    /**
     * Returns true if a valid GGUF file is currently open and parsed.
     */
    bool is_valid() const { return m_is_valid; }

    uint32_t version() const { return m_version; }
    uint64_t tensor_count() const { return m_tensor_count; }
    uint64_t metadata_count() const { return m_metadata_count; }
    uint64_t tensor_data_offset() const { return m_tensor_data_offset; }
    uint32_t alignment() const { return m_alignment; }

    const std::vector<GgufTensorInfo>& tensors() const { return m_tensors; }
    const std::unordered_map<std::string, std::string>& metadata() const { return m_metadata_strings; }

    /**
     * Finds tensor metadata by name.
     */
    const GgufTensorInfo* find_tensor(const std::string& name) const;

    /**
     * Exports parsed GGUF model metadata and tensor catalog formatted as JSON.
     */
    std::string to_json_summary() const;

    /**
     * Helper to write a valid synthetic GGUF model file with quantized tensor layers
     * for verification, benchmark tests, or development without external 5GB downloads.
     */
    static bool create_test_gguf_file(
        const std::string& file_path,
        int num_layers = 4,
        size_t elements_per_layer = 1024 * 256, // 256k elements per layer
        GgmlType qtype = GgmlType::Q4_0
    );

private:
    bool m_is_valid = false;
    std::string m_file_path;
    uint32_t m_version = 0;
    uint64_t m_tensor_count = 0;
    uint64_t m_metadata_count = 0;
    uint32_t m_alignment = 32;
    uint64_t m_tensor_data_offset = 0;

    std::vector<GgufTensorInfo> m_tensors;
    std::unordered_map<std::string, size_t> m_tensor_index;
    std::unordered_map<std::string, std::string> m_metadata_strings;
};

} // namespace streamvex

#endif // GGUF_READER_HPP
