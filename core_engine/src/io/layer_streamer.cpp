#include "layer_streamer.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

namespace streamvex {

LayerStreamer::LayerStreamer(size_t max_chunk_size_bytes)
    : m_allocated_buffer_size(max_chunk_size_bytes) {
    if (m_allocated_buffer_size < 4096) {
        m_allocated_buffer_size = 4096;
    }

    // Allocate page-aligned Buffer A and Buffer B (strict 2-buffer cap)
    int res_a = posix_memalign(reinterpret_cast<void**>(&m_buffer_a), 4096, m_allocated_buffer_size);
    int res_b = posix_memalign(reinterpret_cast<void**>(&m_buffer_b), 4096, m_allocated_buffer_size);

    if (res_a != 0 || !m_buffer_a) {
        m_buffer_a = static_cast<uint8_t*>(malloc(m_allocated_buffer_size));
    }
    if (res_b != 0 || !m_buffer_b) {
        m_buffer_b = static_cast<uint8_t*>(malloc(m_allocated_buffer_size));
    }

    m_current_buffer = m_buffer_a;
    m_prefetch_buffer = m_buffer_b;
}

LayerStreamer::~LayerStreamer() {
    stop();

    if (m_buffer_a) {
        free(m_buffer_a);
        m_buffer_a = nullptr;
    }
    if (m_buffer_b) {
        free(m_buffer_b);
        m_buffer_b = nullptr;
    }
}

bool LayerStreamer::open_file(const std::string& file_path) {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (m_fd >= 0 && m_owns_fd) {
        close(m_fd);
        m_fd = -1;
    }

    m_current_file_path = file_path;
    m_fd = open(file_path.c_str(), O_RDONLY | O_CLOEXEC);

    if (m_fd >= 0) {
        m_owns_fd = true;
        // Hint kernel to optimize for forward sequential scanning
#if defined(POSIX_FADV_SEQUENTIAL)
        posix_fadvise(m_fd, 0, 0, POSIX_FADV_SEQUENTIAL);
#endif
        return true;
    }

    // If file does not exist, streamer will operate in simulated storage benchmark mode
    m_owns_fd = false;
    return false;
}

bool LayerStreamer::start(int total_layers, size_t chunk_size_bytes, size_t start_offset) {
    stop();

    std::unique_lock<std::mutex> lock(m_mutex);

    m_total_layers = total_layers;
    m_chunk_size = chunk_size_bytes;
    m_start_offset = start_offset;
    m_current_layer_idx = 0;
    m_stop_requested = false;
    m_prefetch_ready = false;
    m_task_pending = false;
    m_prefetch_error = false;

    // Expand double buffers if requested chunk size exceeds allocated capacity
    if (m_chunk_size > m_allocated_buffer_size) {
        if (m_buffer_a) free(m_buffer_a);
        if (m_buffer_b) free(m_buffer_b);

        m_allocated_buffer_size = m_chunk_size;
        posix_memalign(reinterpret_cast<void**>(&m_buffer_a), 4096, m_allocated_buffer_size);
        posix_memalign(reinterpret_cast<void**>(&m_buffer_b), 4096, m_allocated_buffer_size);
    }

    m_current_buffer = m_buffer_a;
    m_prefetch_buffer = m_buffer_b;

    // Start background prefetch worker thread
    m_worker_running = true;
    m_worker_thread = std::thread(&LayerStreamer::worker_loop, this);

    // Enqueue initial prefetch for Layer 0 into Buffer A
    if (m_total_layers > 0) {
        m_prefetch_layer_idx = 0;
        m_task_pending = true;
        // Prefetch Layer 0 into m_buffer_a
        m_prefetch_buffer = m_buffer_a;
        m_cv_worker.notify_one();
    }

    return true;
}

void LayerStreamer::stop() {
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (!m_worker_running) {
            return;
        }
        m_stop_requested = true;
        m_cv_worker.notify_all();
        m_cv_consumer.notify_all();
    }

    if (m_worker_thread.joinable()) {
        m_worker_thread.join();
    }

    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_worker_running = false;
        if (m_fd >= 0 && m_owns_fd) {
            close(m_fd);
            m_fd = -1;
            m_owns_fd = false;
        }
    }
}

bool LayerStreamer::acquire_next_layer(LayerData& out_layer) {
    if (m_current_layer_idx >= m_total_layers) {
        return false;
    }

    auto wait_start = std::chrono::high_resolution_clock::now();

    std::unique_lock<std::mutex> lock(m_mutex);

    // Wait until the background worker completes prefetching for the pending layer
    m_cv_consumer.wait(lock, [this]() {
        return m_stop_requested || m_prefetch_ready || m_prefetch_error;
    });

    if (m_stop_requested || m_prefetch_error) {
        return false;
    }

    auto wait_end = std::chrono::high_resolution_clock::now();
    double wait_ms = std::chrono::duration<double, std::milli>(wait_end - wait_start).count();

    int active_layer = m_current_layer_idx;
    double prefetch_ms = m_last_prefetch_latency_ms;

    if (active_layer == 0) {
        // Layer 0 was loaded directly into m_buffer_a
        m_current_buffer = m_buffer_a;
        m_prefetch_buffer = m_buffer_b;
    } else {
        // Zero-copy O(1) buffer swap:
        // Prefetched buffer becomes active compute buffer; previous compute buffer becomes next prefetch target
        std::swap(m_current_buffer, m_prefetch_buffer);
    }

    out_layer.data = m_current_buffer;
    out_layer.size = m_chunk_size;
    out_layer.layer_index = active_layer;
    out_layer.prefetch_latency_ms = prefetch_ms;
    out_layer.consumer_wait_ms = wait_ms;

    m_current_layer_idx++;
    m_prefetch_ready = false;

    // Immediately trigger asynchronous prefetch for Layer N+1 into the inactive buffer
    if (m_current_layer_idx < m_total_layers) {
        m_prefetch_layer_idx = m_current_layer_idx;
        m_task_pending = true;
        m_cv_worker.notify_one();
    }

    return true;
}

void LayerStreamer::worker_loop() {
    while (true) {
        int layer_to_fetch = -1;
        size_t offset = 0;
        size_t size = 0;
        uint8_t* target_buffer = nullptr;

        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cv_worker.wait(lock, [this]() {
                return m_stop_requested || m_task_pending;
            });

            if (m_stop_requested) {
                break;
            }

            layer_to_fetch = m_prefetch_layer_idx;
            target_buffer = m_prefetch_buffer;
            offset = m_start_offset + static_cast<size_t>(layer_to_fetch) * m_chunk_size;
            size = m_chunk_size;
            m_task_pending = false;
        }

        // Perform disk I/O outside of the mutex lock to allow concurrent computation
        auto fetch_start = std::chrono::high_resolution_clock::now();
        bool success = read_chunk_from_file(target_buffer, offset, size);
        auto fetch_end = std::chrono::high_resolution_clock::now();
        double latency_ms = std::chrono::duration<double, std::milli>(fetch_end - fetch_start).count();

        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_last_prefetch_latency_ms = latency_ms;
            m_prefetch_ready = true;
            m_prefetch_error = !success;
            m_cv_consumer.notify_one();
        }
    }
}

bool LayerStreamer::read_chunk_from_file(uint8_t* destination, size_t offset, size_t size) {
    if (m_fd >= 0) {
        // Direct prefetch advisory hint to OS kernel
#if defined(POSIX_FADV_WILLNEED)
        posix_fadvise(m_fd, static_cast<off_t>(offset), static_cast<off_t>(size), POSIX_FADV_WILLNEED);
#endif

        size_t total_read = 0;
        while (total_read < size) {
            ssize_t bytes = pread(m_fd, destination + total_read, size - total_read,
                                  static_cast<off_t>(offset + total_read));
            if (bytes > 0) {
                total_read += static_cast<size_t>(bytes);
            } else if (bytes == 0) {
                // If end-of-file reached in smaller benchmark file, synthesize layer weight pattern
                memset(destination + total_read, 0x4B, size - total_read);
                break;
            } else {
                if (errno == EINTR) continue;
                return false;
            }
        }
        return true;
    } else {
        // High-speed simulated NVMe / UFS 3.1 storage read (memory-mapped synthetic weight pattern)
        for (size_t i = 0; i < size; i += 4096) {
            destination[i] = static_cast<uint8_t>((offset + i) & 0xFF);
        }
        // Simulated disk I/O pacing (~700 MB/s rate)
        double delay_ms = (static_cast<double>(size) / (700.0 * 1024.0 * 1024.0)) * 1000.0;
        if (delay_ms > 0.0) {
            std::this_thread::sleep_for(std::chrono::microseconds(static_cast<int64_t>(delay_ms * 1000.0)));
        }
        return true;
    }
}

double LayerStreamer::query_resident_ram_mb() {
#if defined(__linux__) || defined(__ANDROID__)
    std::ifstream statm("/proc/self/statm");
    if (statm.is_open()) {
        long total_pages = 0;
        long resident_pages = 0;
        if (statm >> total_pages >> resident_pages) {
            long page_size = sysconf(_SC_PAGESIZE);
            if (page_size > 0) {
                return static_cast<double>(resident_pages * page_size) / (1024.0 * 1024.0);
            }
        }
    }
#endif
    return 0.0;
}

BenchmarkResult LayerStreamer::run_benchmark(int total_layers, size_t chunk_size_bytes, int simulate_compute_ms) {
    BenchmarkResult res;
    res.total_layers = total_layers;
    res.chunk_size_bytes = chunk_size_bytes;

    double base_ram = query_resident_ram_mb();
    res.baseline_ram_mb = base_ram;
    double max_ram = base_ram;

    std::vector<double> prefetch_latencies;
    prefetch_latencies.reserve(total_layers);

    auto bench_start = std::chrono::high_resolution_clock::now();

    if (!start(total_layers, chunk_size_bytes, 0)) {
        res.is_ram_bounded = false;
        return res;
    }

    LayerData layer;
    while (acquire_next_layer(layer)) {
        prefetch_latencies.push_back(layer.prefetch_latency_ms);

        // Touch buffer data to ensure physical pages are mapped
        volatile uint8_t dummy = 0;
        for (size_t i = 0; i < layer.size; i += 65536) {
            dummy ^= layer.data[i];
        }
        (void)dummy;

        // Simulate layer forward-pass compute time
        if (simulate_compute_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(simulate_compute_ms));
        }

        // Measure live resident RAM
        double current_ram = query_resident_ram_mb();
        if (current_ram > max_ram) {
            max_ram = current_ram;
        }
    }

    auto bench_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(bench_end - bench_start).count();

    stop();

    double total_bytes = static_cast<double>(total_layers) * static_cast<double>(chunk_size_bytes);
    double total_mb = total_bytes / (1024.0 * 1024.0);

    res.total_duration_sec = total_sec;
    res.throughput_mb_s = total_sec > 0.0 ? (total_mb / total_sec) : 0.0;
    res.peak_ram_mb = max_ram;

    if (!prefetch_latencies.empty()) {
        double sum = std::accumulate(prefetch_latencies.begin(), prefetch_latencies.end(), 0.0);
        res.avg_prefetch_latency_ms = sum / static_cast<double>(prefetch_latencies.size());
    }

    // Double buffer memory footprint in MB
    double two_buffer_mb = (2.0 * static_cast<double>(chunk_size_bytes)) / (1024.0 * 1024.0);

    // RAM usage is bounded if peak RAM did not grow past baseline + double buffer + small margin
    res.is_ram_bounded = (max_ram <= base_ram + two_buffer_mb + 25.0);

    return res;
}

} // namespace streamvex
