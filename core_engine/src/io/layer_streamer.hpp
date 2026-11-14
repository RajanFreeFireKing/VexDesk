#ifndef LAYER_STREAMER_HPP
#define LAYER_STREAMER_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>

namespace streamvex {

/**
 * Metadata and pointer to an active model layer chunk in memory.
 */
struct LayerData {
    const uint8_t* data = nullptr;
    size_t size = 0;
    int layer_index = -1;
    double prefetch_latency_ms = 0.0;
    double consumer_wait_ms = 0.0;
};

/**
 * Summary metrics of a layer streaming benchmark run.
 */
struct BenchmarkResult {
    int total_layers = 0;
    size_t chunk_size_bytes = 0;
    double total_duration_sec = 0.0;
    double throughput_mb_s = 0.0;
    double avg_prefetch_latency_ms = 0.0;
    double peak_ram_mb = 0.0;
    double baseline_ram_mb = 0.0;
    bool is_ram_bounded = true;
};

/**
 * Asynchronous Double-Buffering Layer Streamer.
 * 
 * Streams large model layers sequentially from disk chunk-by-chunk so that
 * system RAM usage is strictly capped to two layer buffers (Buffer A & Buffer B)
 * at any time. While the compute thread operates on Buffer A, a background
 * worker thread prefetches Layer N+1 into Buffer B, swapping pointers in O(1)
 * zero-copy time upon completion.
 */
class LayerStreamer {
public:
    explicit LayerStreamer(size_t max_chunk_size_bytes = 64 * 1024 * 1024);
    ~LayerStreamer();

    // Prevent copies to ensure RAII double-buffer integrity
    LayerStreamer(const LayerStreamer&) = delete;
    LayerStreamer& operator=(const LayerStreamer&) = delete;

    /**
     * Opens a model file for asynchronous streaming.
     * Applies POSIX sequential and willneed hints to minimize kernel readahead overhead.
     */
    bool open_file(const std::string& file_path);

    /**
     * Initializes the streaming pipeline for the specified number of layers and chunk size.
     * Launches the background prefetch worker and initiates prefetching for Layer 0.
     */
    bool start(int total_layers, size_t chunk_size_bytes, size_t start_offset = 0);

    /**
     * Acquires the next layer buffer for the compute thread.
     * Swaps double-buffers in O(1) time and dispatches background prefetch for Layer N+1.
     * 
     * @param out_layer Populated with layer buffer pointer, index, and timing stats.
     * @return true if a layer was successfully acquired, false if all layers are finished.
     */
    bool acquire_next_layer(LayerData& out_layer);

    /**
     * Gracefully stops the worker thread and closes file descriptors.
     */
    void stop();

    /**
     * Executes an end-to-end streaming benchmark over simulated or real model layers.
     * Measures disk throughput (MB/s), average prefetch latency (ms), and peak resident RAM.
     */
    BenchmarkResult run_benchmark(int total_layers, size_t chunk_size_bytes, int simulate_compute_ms = 2);

    /**
     * Queries the process resident set size (RSS) in Megabytes from /proc/self/statm.
     */
    static double query_resident_ram_mb();

private:
    void worker_loop();
    bool read_chunk_from_file(uint8_t* destination, size_t offset, size_t size);

    // Double buffers (Buffer A & Buffer B)
    uint8_t* m_buffer_a = nullptr;
    uint8_t* m_buffer_b = nullptr;
    size_t m_allocated_buffer_size = 0;

    // Active pointers for O(1) pointer swap
    uint8_t* m_current_buffer = nullptr;
    uint8_t* m_prefetch_buffer = nullptr;

    // File descriptor
    int m_fd = -1;
    bool m_owns_fd = false;
    std::string m_current_file_path;

    // Pipeline configuration
    int m_total_layers = 0;
    size_t m_chunk_size = 0;
    size_t m_start_offset = 0;
    int m_current_layer_idx = 0;

    // Thread synchronization
    std::thread m_worker_thread;
    std::mutex m_mutex;
    std::condition_variable m_cv_worker;
    std::condition_variable m_cv_consumer;

    // Worker state
    bool m_worker_running = false;
    bool m_stop_requested = false;
    int m_prefetch_layer_idx = -1;
    bool m_task_pending = false;
    bool m_prefetch_ready = false;
    double m_last_prefetch_latency_ms = 0.0;
    bool m_prefetch_error = false;
};

} // namespace streamvex

#endif // LAYER_STREAMER_HPP
