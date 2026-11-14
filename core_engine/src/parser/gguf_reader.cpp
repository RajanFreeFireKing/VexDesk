#include "gguf_reader.hpp"
#include "../compute/dequant.hpp"

#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include <iomanip>
#include <algorithm>

namespace streamvex {

const char* ggml_type_name(GgmlType type) {
    switch (type) {
        case GgmlType::F32:  return "F32";
        case GgmlType::F16:  return "F16";
        case GgmlType::Q4_0: return "Q4_0";
        case GgmlType::Q4_1: return "Q4_1";
        case GgmlType::Q5_0: return "Q5_0";
        case GgmlType::Q5_1: return "Q5_1";
        case GgmlType::Q8_0: return "Q8_0";
        case GgmlType::Q8_1: return "Q8_1";
        case GgmlType::Q2_K: return "Q2_K";
        case GgmlType::Q3_K: return "Q3_K";
        case GgmlType::Q4_K: return "Q4_K";
        case GgmlType::Q5_K: return "Q5_K";
        case GgmlType::Q6_K: return "Q6_K";
        case GgmlType::Q8_K: return "Q8_K";
        default:             return "UNKNOWN";
    }
}

size_t ggml_type_block_size(GgmlType type) {
    switch (type) {
        case GgmlType::Q4_0:
        case GgmlType::Q4_1:
        case GgmlType::Q5_0:
        case GgmlType::Q5_1:
        case GgmlType::Q8_0:
        case GgmlType::Q8_1:
            return 32;
        case GgmlType::Q2_K:
        case GgmlType::Q3_K:
        case GgmlType::Q4_K:
        case GgmlType::Q5_K:
        case GgmlType::Q6_K:
        case GgmlType::Q8_K:
            return 256;
        default:
            return 1;
    }
}

size_t ggml_type_type_size(GgmlType type) {
    switch (type) {
        case GgmlType::F32:  return sizeof(float);
        case GgmlType::F16:  return sizeof(uint16_t);
        case GgmlType::Q4_0: return sizeof(block_q4_0); // 18 bytes for 32 elements
        case GgmlType::Q4_K: return sizeof(block_q4_K); // 144 bytes for 256 elements
        default:             return sizeof(float);
    }
}

GgufReader::GgufReader() = default;
GgufReader::~GgufReader() = default;

static bool read_gguf_string(std::ifstream& fin, std::string& out_str) {
    uint64_t len = 0;
    if (!fin.read(reinterpret_cast<char*>(&len), sizeof(len))) return false;
    if (len > 1024 * 1024) return false; // Sanity guard
    out_str.resize(len);
    if (len > 0) {
        if (!fin.read(&out_str[0], len)) return false;
    }
    return true;
}

static bool skip_gguf_value(std::ifstream& fin, GgufValueType type) {
    switch (type) {
        case GgufValueType::UINT8:
        case GgufValueType::INT8:
        case GgufValueType::BOOL:
            fin.seekg(1, std::ios::cur);
            break;
        case GgufValueType::UINT16:
        case GgufValueType::INT16:
            fin.seekg(2, std::ios::cur);
            break;
        case GgufValueType::UINT32:
        case GgufValueType::INT32:
        case GgufValueType::FLOAT32:
            fin.seekg(4, std::ios::cur);
            break;
        case GgufValueType::UINT64:
        case GgufValueType::INT64:
        case GgufValueType::FLOAT64:
            fin.seekg(8, std::ios::cur);
            break;
        case GgufValueType::STRING: {
            uint64_t len = 0;
            if (!fin.read(reinterpret_cast<char*>(&len), sizeof(len))) return false;
            fin.seekg(len, std::ios::cur);
            break;
        }
        case GgufValueType::ARRAY: {
            uint32_t elem_type_raw = 0;
            uint64_t arr_len = 0;
            if (!fin.read(reinterpret_cast<char*>(&elem_type_raw), sizeof(elem_type_raw))) return false;
            if (!fin.read(reinterpret_cast<char*>(&arr_len), sizeof(arr_len))) return false;
            auto elem_type = static_cast<GgufValueType>(elem_type_raw);
            for (uint64_t i = 0; i < arr_len; ++i) {
                if (!skip_gguf_value(fin, elem_type)) return false;
            }
            break;
        }
        default:
            return false;
    }
    return true;
}

bool GgufReader::open(const std::string& file_path) {
    m_is_valid = false;
    m_file_path = file_path;
    m_tensors.clear();
    m_tensor_index.clear();
    m_metadata_strings.clear();

    std::ifstream fin(file_path, std::ios::binary);
    if (!fin.is_open()) {
        return false;
    }

    // 1. Check Magic Number ("GGUF" = 0x46554747)
    char magic[4] = {0};
    if (!fin.read(magic, 4) || std::memcmp(magic, "GGUF", 4) != 0) {
        return false;
    }

    // 2. Read Version (v2 or v3)
    if (!fin.read(reinterpret_cast<char*>(&m_version), sizeof(m_version))) return false;
    if (m_version < 2 || m_version > 3) {
        return false;
    }

    // 3. Tensor count & Metadata KV count
    if (!fin.read(reinterpret_cast<char*>(&m_tensor_count), sizeof(m_tensor_count))) return false;
    if (!fin.read(reinterpret_cast<char*>(&m_metadata_count), sizeof(m_metadata_count))) return false;

    // 4. Parse Metadata KV pairs
    for (uint64_t i = 0; i < m_metadata_count; ++i) {
        std::string key;
        if (!read_gguf_string(fin, key)) return false;

        uint32_t val_type_raw = 0;
        if (!fin.read(reinterpret_cast<char*>(&val_type_raw), sizeof(val_type_raw))) return false;
        auto val_type = static_cast<GgufValueType>(val_type_raw);

        if (val_type == GgufValueType::STRING) {
            std::string str_val;
            if (!read_gguf_string(fin, str_val)) return false;
            m_metadata_strings[key] = str_val;
        } else if (val_type == GgufValueType::UINT32) {
            uint32_t u32 = 0;
            if (!fin.read(reinterpret_cast<char*>(&u32), sizeof(u32))) return false;
            m_metadata_strings[key] = std::to_string(u32);
            if (key == "general.alignment") {
                m_alignment = u32;
            }
        } else if (val_type == GgufValueType::UINT64) {
            uint64_t u64 = 0;
            if (!fin.read(reinterpret_cast<char*>(&u64), sizeof(u64))) return false;
            m_metadata_strings[key] = std::to_string(u64);
        } else if (val_type == GgufValueType::INT32) {
            int32_t i32 = 0;
            if (!fin.read(reinterpret_cast<char*>(&i32), sizeof(i32))) return false;
            m_metadata_strings[key] = std::to_string(i32);
        } else if (val_type == GgufValueType::FLOAT32) {
            float f = 0.0f;
            if (!fin.read(reinterpret_cast<char*>(&f), sizeof(f))) return false;
            m_metadata_strings[key] = std::to_string(f);
        } else {
            if (!skip_gguf_value(fin, val_type)) return false;
        }
    }

    // 5. Parse Tensor Info Catalog
    m_tensors.reserve(m_tensor_count);
    for (uint64_t i = 0; i < m_tensor_count; ++i) {
        GgufTensorInfo info;
        if (!read_gguf_string(fin, info.name)) return false;

        if (!fin.read(reinterpret_cast<char*>(&info.n_dimensions), sizeof(info.n_dimensions))) return false;
        if (info.n_dimensions > 8) return false; // Sanity check

        info.dimensions.resize(info.n_dimensions);
        info.num_elements = 1;
        for (uint32_t d = 0; d < info.n_dimensions; ++d) {
            if (!fin.read(reinterpret_cast<char*>(&info.dimensions[d]), sizeof(uint64_t))) return false;
            info.num_elements *= info.dimensions[d];
        }

        uint32_t type_raw = 0;
        if (!fin.read(reinterpret_cast<char*>(&type_raw), sizeof(type_raw))) return false;
        info.type = static_cast<GgmlType>(type_raw);

        if (!fin.read(reinterpret_cast<char*>(&info.tensor_offset), sizeof(info.tensor_offset))) return false;

        // Calculate size on disk based on quantization type
        size_t block_size = ggml_type_block_size(info.type);
        size_t type_size = ggml_type_type_size(info.type);
        info.size_bytes = (info.num_elements / block_size) * type_size;

        m_tensor_index[info.name] = m_tensors.size();
        m_tensors.push_back(info);
    }

    // 6. Calculate Tensor Data Start Offset (Aligned to m_alignment)
    uint64_t current_file_pos = fin.tellg();
    uint64_t align = m_alignment > 0 ? m_alignment : 32;
    m_tensor_data_offset = (current_file_pos + (align - 1)) & ~(align - 1);

    // Compute absolute file offsets for zero-copy streaming
    for (auto& t : m_tensors) {
        t.file_offset = m_tensor_data_offset + t.tensor_offset;
    }

    m_is_valid = true;
    return true;
}

const GgufTensorInfo* GgufReader::find_tensor(const std::string& name) const {
    auto it = m_tensor_index.find(name);
    if (it != m_tensor_index.end()) {
        return &m_tensors[it->second];
    }
    return nullptr;
}

std::string GgufReader::to_json_summary() const {
    std::ostringstream json;
    json << "{\n";
    json << "  \"valid\": " << (m_is_valid ? "true" : "false") << ",\n";
    json << "  \"version\": " << m_version << ",\n";
    json << "  \"tensor_count\": " << m_tensor_count << ",\n";
    json << "  \"metadata_count\": " << m_metadata_count << ",\n";
    json << "  \"alignment\": " << m_alignment << ",\n";
    json << "  \"tensor_data_offset\": " << m_tensor_data_offset << ",\n";
    json << "  \"metadata\": {\n";

    size_t count = 0;
    for (const auto& kv : m_metadata_strings) {
        json << "    \"" << kv.first << "\": \"" << kv.second << "\"";
        if (++count < m_metadata_strings.size()) json << ",";
        json << "\n";
    }
    json << "  },\n";

    json << "  \"tensors\": [\n";
    for (size_t i = 0; i < m_tensors.size(); ++i) {
        const auto& t = m_tensors[i];
        json << "    {\n";
        json << "      \"name\": \"" << t.name << "\",\n";
        json << "      \"type\": \"" << ggml_type_name(t.type) << "\",\n";
        json << "      \"dimensions\": [";
        for (size_t d = 0; d < t.dimensions.size(); ++d) {
            json << t.dimensions[d] << (d + 1 < t.dimensions.size() ? ", " : "");
        }
        json << "],\n";
        json << "      \"num_elements\": " << t.num_elements << ",\n";
        json << "      \"size_bytes\": " << t.size_bytes << ",\n";
        json << "      \"file_offset\": " << t.file_offset << "\n";
        json << "    }" << (i + 1 < m_tensors.size() ? "," : "") << "\n";
    }
    json << "  ]\n";
    json << "}";
    return json.str();
}

bool GgufReader::create_test_gguf_file(
    const std::string& file_path,
    int num_layers,
    size_t elements_per_layer,
    GgmlType qtype
) {
    std::ofstream fout(file_path, std::ios::binary);
    if (!fout.is_open()) return false;

    // 1. Magic & Version
    fout.write("GGUF", 4);
    uint32_t version = 3;
    fout.write(reinterpret_cast<const char*>(&version), sizeof(version));

    // 2. Counts
    uint64_t n_tensors = num_layers;
    uint64_t n_metadata = 4;
    fout.write(reinterpret_cast<const char*>(&n_tensors), sizeof(n_tensors));
    fout.write(reinterpret_cast<const char*>(&n_metadata), sizeof(n_metadata));

    // Helper lambda for writing GGUF string
    auto write_str = [&](const std::string& s) {
        uint64_t len = s.size();
        fout.write(reinterpret_cast<const char*>(&len), sizeof(len));
        fout.write(s.data(), len);
    };

    // 3. Write metadata KV
    write_str("general.architecture");
    uint32_t str_type = static_cast<uint32_t>(GgufValueType::STRING);
    fout.write(reinterpret_cast<const char*>(&str_type), sizeof(str_type));
    write_str("streamvex-local");

    write_str("general.name");
    fout.write(reinterpret_cast<const char*>(&str_type), sizeof(str_type));
    write_str("VexDesk-Test-Q4");

    write_str("general.alignment");
    uint32_t u32_type = static_cast<uint32_t>(GgufValueType::UINT32);
    fout.write(reinterpret_cast<const char*>(&u32_type), sizeof(u32_type));
    uint32_t align = 32;
    fout.write(reinterpret_cast<const char*>(&align), sizeof(align));

    write_str("streamvex.context_length");
    fout.write(reinterpret_cast<const char*>(&u32_type), sizeof(u32_type));
    uint32_t ctx_len = 2048;
    fout.write(reinterpret_cast<const char*>(&ctx_len), sizeof(ctx_len));

    // Calculate bytes per layer tensor
    size_t block_size = ggml_type_block_size(qtype);
    size_t type_size = ggml_type_type_size(qtype);
    size_t tensor_bytes = (elements_per_layer / block_size) * type_size;

    // 4. Write Tensor Infos
    uint64_t current_tensor_offset = 0;
    for (int l = 0; l < num_layers; ++l) {
        std::string tname = "blk." + std::to_string(l) + ".attn_q.weight";
        write_str(tname);

        uint32_t dims = 2;
        fout.write(reinterpret_cast<const char*>(&dims), sizeof(dims));
        uint64_t d0 = elements_per_layer / 1024;
        uint64_t d1 = 1024;
        fout.write(reinterpret_cast<const char*>(&d0), sizeof(d0));
        fout.write(reinterpret_cast<const char*>(&d1), sizeof(d1));

        uint32_t raw_type = static_cast<uint32_t>(qtype);
        fout.write(reinterpret_cast<const char*>(&raw_type), sizeof(raw_type));
        fout.write(reinterpret_cast<const char*>(&current_tensor_offset), sizeof(current_tensor_offset));

        current_tensor_offset += tensor_bytes;
    }

    // 5. Pad to alignment boundary
    uint64_t cur_pos = fout.tellp();
    uint64_t aligned_start = (cur_pos + (align - 1)) & ~(align - 1);
    while (cur_pos < aligned_start) {
        char zero = 0;
        fout.write(&zero, 1);
        cur_pos++;
    }

    // 6. Generate and write quantized weights
    std::vector<float> sample_floats(elements_per_layer);
    for (size_t i = 0; i < elements_per_layer; ++i) {
        sample_floats[i] = std::sin(static_cast<float>(i) * 0.05f) * 1.5f;
    }

    std::vector<uint8_t> quantized_buffer(tensor_bytes);
    if (qtype == GgmlType::Q4_0) {
        quantize_row_q4_0(sample_floats.data(), quantized_buffer.data(), elements_per_layer);
    } else {
        quantize_row_q4_K(sample_floats.data(), quantized_buffer.data(), elements_per_layer);
    }

    for (int l = 0; l < num_layers; ++l) {
        fout.write(reinterpret_cast<const char*>(quantized_buffer.data()), quantized_buffer.size());
    }

    return true;
}

} // namespace streamvex
