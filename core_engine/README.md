# StreamVex C++ Inference Engine

**StreamVex** is an ultra-lightweight, high-performance C++ LLM inference engine engineered specifically for edge and mobile architectures (ARM64 / Android / Linux / macOS). It features asynchronous double-buffering layer streaming, ARM64 NEON SIMD acceleration, zero-copy GGUF metadata parsing, and an `extern "C"` foreign-function interface (FFI) ready for Flutter, Android JNI, Swift, Python, and native C++ applications.

---

## 🚀 Key Architectural Features

- **Asynchronous Double-Buffering IO:** Overlaps layer weight disk reads with matrix computation on the CPU/GPU, eliminating disk IO stalls during autoregressive generation.
- **ARM64 NEON SIMD Optimization:** Highly tuned kernels for Q4_0, Q4_K_M, and Q8_0 dequantization, vector dot products, and GEMV matrix multiplication.
- **Direct GGUF Binary Parser:** Parses GGUF v2/v3 header structures, tensor metadata catalogues, and quantization block headers directly without loading entire model weights into RAM.
- **Zero-Friction C-ABI (`extern "C"`):** Standard C calling convention export for clean integration into Flutter (`dart:ffi`), Android NDK, Python (`ctypes`), Rust, and C/C++.
- **Integrated llama.cpp Backend:** Leverages battle-tested `ggml` backend primitives for maximum hardware compatibility and instruction set coverage.

---

## 📁 Source Directory Structure

```text
core_engine/
├── CMakeLists.txt              # Standard cross-platform CMake build configuration
├── Makefile                    # Direct Makefile for Termux, Linux, and Android NDK
├── README.md                   # Engine documentation and API reference
├── include/
│   └── streamvex.h             # Public C-API header declarations (extern "C")
├── src/
│   ├── streamvex.cpp           # Main engine lifecycle and streaming runtime
│   ├── jni_bridge.cpp          # Android JNI bindings for NativeBridge Java/Kotlin
│   ├── io/
│   │   ├── layer_streamer.hpp  # Asynchronous double-buffered layer prefetcher
│   │   └── layer_streamer.cpp
│   ├── parser/
│   │   ├── gguf_reader.hpp     # Zero-copy GGUF binary format parser
│   │   └── gguf_reader.cpp
│   ├── compute/
│   │   ├── dequant.hpp         # ARM64 NEON SIMD dequantization kernels
│   │   ├── dequant.cpp
│   │   ├── gemv.hpp            # High-throughput Matrix-Vector kernels
│   │   ├── gemv.cpp
│   │   ├── transformer_layer.hpp # Transformer block computation & RMSNorm
│   │   ├── transformer_layer.cpp
│   │   ├── sampler.hpp         # Top-K, Top-P, Temperature token sampling
│   │   └── sampler.cpp
│   └── engine/
│       ├── autoregressive_engine.hpp # Full context generation pipeline
│       └── autoregressive_engine.cpp
├── third_party/
│   └── llama.cpp/              # Core ggml & llama runtime dependencies
├── test_ffi_runner.cpp         # Standalone FFI verification executable
└── test_llama_load.cpp         # Standalone model load verification
```

---

## 📚 C-API Function Reference (`include/streamvex.h`)

All API functions export standard C linkage (`extern "C"`) and are declared in [`include/streamvex.h`](include/streamvex.h).

### 1. `init_engine`
Initializes the engine and loads the model from the specified `.gguf` file path.
```c
STREAMVEX_API int init_engine(const char* model_path);
```
- **Parameters:**
  - `model_path`: Null-terminated string containing the absolute file path to the `.gguf` model file.
- **Returns:** `0` on success, or a non-zero error code if loading fails.

---

### 2. `stream_token`
Streams tokens autoregressively for the given prompt, invoking the callback function for every generated token chunk.
```c
typedef void (*TokenCallback)(const char* token);

STREAMVEX_API int stream_token(const char* prompt, TokenCallback callback);
```
- **Parameters:**
  - `prompt`: Null-terminated user prompt string.
  - `callback`: Function pointer called sequentially with each generated token string.
- **Returns:** `0` on successful completion, non-zero if generation encounters an error.

---

### 3. `get_memory_usage`
Retrieves current resident memory (RSS RAM) consumed by the engine in Megabytes (MB).
```c
STREAMVEX_API double get_memory_usage(void);
```
- **Returns:** Double-precision float representing current RAM usage in MB.

---

### 4. `parse_gguf_metadata`
Inspects a `.gguf` model file on disk and extracts architecture type, parameter counts, context length, tensor counts, and layer layouts into a JSON string without loading tensor weights.
```c
STREAMVEX_API int parse_gguf_metadata(
    const char* model_path,
    char* out_json_info,
    size_t max_len
);
```
- **Parameters:**
  - `model_path`: Path to the `.gguf` model file.
  - `out_json_info`: Caller-allocated character buffer to receive the formatted JSON string.
  - `max_len`: Capacity of `out_json_info` buffer in bytes.
- **Returns:** `0` on success, non-zero on error.

---

### 5. `stream_and_dequant_layer`
Streams a specific layer tensor from disk and dequantizes its weights into a float32 array using ARM64 NEON SIMD kernels.
```c
STREAMVEX_API int stream_and_dequant_layer(
    const char* model_path,
    const char* tensor_name,
    float* out_buffer,
    size_t max_floats
);
```
- **Parameters:**
  - `model_path`: Path to the `.gguf` file.
  - `tensor_name`: Name of the tensor in the GGUF catalog (e.g. `"blk.0.attn_q.weight"`).
  - `out_buffer`: Pointer to pre-allocated `float` array.
  - `max_floats`: Maximum number of `float` elements `out_buffer` can hold.
- **Returns:** Number of dequantized floats on success, negative error code on failure.

---

### 6. `compute_gemv_benchmark`
Runs matrix-vector multiplication benchmarks comparing float32 vs quantized Q4_0 kernels and outputs latency and GFLOPS in JSON.
```c
STREAMVEX_API int compute_gemv_benchmark(
    int rows,
    int cols,
    char* out_json_stats,
    size_t max_len
);
```

---

### 7. `run_streaming_benchmark`
Simulates double-buffering sequential layer disk streaming, measuring sustained transfer bandwidth (MB/s) and frame pacing.
```c
STREAMVEX_API int run_streaming_benchmark(
    const char* file_path,
    size_t chunk_size_mb,
    int total_layers,
    char* out_json_stats,
    size_t max_len
);
```

---

### 8. `is_engine_ready`
Checks whether a model is loaded and ready to accept inference requests.
```c
STREAMVEX_API int is_engine_ready(void);
```
- **Returns:** `1` if loaded and ready, `0` otherwise.

---

### 9. `get_last_generation_telemetry`
Returns detailed performance metrics (Time-to-First-Token in ms, tokens/sec, total tokens, peak RAM) from the last generation run as a JSON object.
```c
STREAMVEX_API int get_last_generation_telemetry(char* out_json, size_t max_len);
```

---

### 10. `release_engine`
Releases model weights, frees KV cache allocations, and resets engine state.
```c
STREAMVEX_API void release_engine(void);
```

---

## 🔨 Build Instructions

### Option A: Direct Build with `make` (Linux / Termux)
To compile `libstreamvex.so` directly using Clang or GCC:
```bash
cd core_engine
make -j$(nproc)
```
To clean build artifacts:
```bash
make clean
```

### Option B: Build with `cmake` (Cross-Platform)
```bash
cd core_engine
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

### Option C: Android NDK Cross-Compilation (ARM64-v8a)
To cross-compile for Android devices using the Android NDK:
```bash
cd core_engine
mkdir -p build_android && cd build_android
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK_HOME}/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-24 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```
The compiled `libstreamvex.so` will be generated in `build_android/` and can be placed in `android/app/src/main/jniLibs/arm64-v8a/`.

### Option D: Building and Running the Test Runner
```bash
cd core_engine
clang++ -std=c++17 -O3 -Iinclude test_ffi_runner.cpp -L. -lstreamvex -Wl,-rpath,. -o test_ffi_runner
./test_ffi_runner
```

---

## 📄 License
StreamVex core engine is released under the Apache-2.0 / MIT License. Embedded `llama.cpp` / `ggml` components are licensed under their respective MIT upstream licenses.
