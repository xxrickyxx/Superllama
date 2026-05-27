#pragma once
#include "common/platform.h"

#include "common/types.h"
#include <windows.h>
#include <string>
#include <vector>
#include <queue>
#include <thread>
#include <atomic>
#include <functional>

namespace sl {

// ====================================================================
// AsyncSSDStreamer - high-performance overlapped I/O engine
// 
// Provides:
// - Ring-buffer based streaming reads
// - Batch I/O submission
// - Priority-based queue
// - I/O completion callbacks
// - Throughput measurement
// ====================================================================
class AsyncSSDStreamer {
public:
    // Callback when a chunk is read and ready in memory
    using CompletionCallback = std::function<void(uint64_t offset, void* data, uint64_t size, bool success)>;
    
    struct ReadRequest {
        uint64_t offset;
        uint64_t size;
        void* buffer;
        int priority;
        CompletionCallback on_complete;
        uint64_t request_id;
    };
    
    AsyncSSDStreamer();
    ~AsyncSSDStreamer();
    
    // Initialize with a file
    bool open(const std::string& path);
    void close();
    
    // Submit async read requests
    // Returns immediately; completion is delivered via callback
    void submit_read(ReadRequest&& request);
    
    // Submit batch of reads
    void submit_batch(std::vector<ReadRequest>&& requests);
    
    // Wait for all pending I/O to complete
    void flush();
    
    // Statistics
    uint64_t total_bytes_read() const { return total_bytes_read_; }
    double throughput_mbps() const;
    uint64_t pending_io_count() const { return pending_count_; }
    
private:
    HANDLE file_handle_ = INVALID_HANDLE_VALUE;
    HANDLE iocp_ = nullptr;
    
    std::thread completion_thread_;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> pending_count_{0};
    
    std::atomic<uint64_t> total_bytes_read_{0};
    std::atomic<uint64_t> request_counter_{0};
    
    void completion_worker();
    
    struct OverlappedContext {
        OVERLAPPED overlapped;
        CompletionCallback callback;
        uint64_t request_id;
        void* buffer;
        uint64_t size;
    };
};

} // namespace sl