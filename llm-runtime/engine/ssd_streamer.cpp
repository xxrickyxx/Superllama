#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif
#include "ssd_streamer.h"
#include "common/logging.h"
#include <chrono>

namespace sl {

AsyncSSDStreamer::AsyncSSDStreamer() {
    iocp_ = CreateIoCompletionPort(INVALID_HANDLE_VALUE, nullptr, 0, 0);
}

AsyncSSDStreamer::~AsyncSSDStreamer() {
    close();
    if (iocp_) CloseHandle(iocp_);
}

bool AsyncSSDStreamer::open(const std::string& path) {
    file_handle_ = CreateFileA(
        path.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        FILE_FLAG_OVERLAPPED | FILE_FLAG_NO_BUFFERING,  // Unbuffered for max throughput
        nullptr
    );
    
    if (file_handle_ == INVALID_HANDLE_VALUE) {
        DWORD err = GetLastError();
        SL_LOG_WARN("Unbuffered open failed ({}), trying buffered...", err);
        
        // Fallback to buffered
        file_handle_ = CreateFileA(
            path.c_str(),
            GENERIC_READ,
            FILE_SHARE_READ,
            nullptr,
            OPEN_EXISTING,
            FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN,
            nullptr
        );
        
        if (file_handle_ == INVALID_HANDLE_VALUE) {
            SL_LOG_ERROR("Failed to open file: {}", path);
            return false;
        }
    }
    
    // Associate with IOCP
    CreateIoCompletionPort(file_handle_, iocp_, 0, 0);
    
    // Start completion worker
    running_ = true;
    completion_thread_ = std::thread(&AsyncSSDStreamer::completion_worker, this);
    
    return true;
}

void AsyncSSDStreamer::close() {
    running_ = false;
    if (completion_thread_.joinable()) {
        completion_thread_.join();
    }
    if (file_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
    }
}

void AsyncSSDStreamer::submit_read(ReadRequest&& request) {
    auto* ctx = new OverlappedContext();
    ctx->callback = std::move(request.on_complete);
    ctx->request_id = request.request_id;
    ctx->buffer = request.buffer;
    ctx->size = request.size;
    
    memset(&ctx->overlapped, 0, sizeof(OVERLAPPED));
    ctx->overlapped.Offset = static_cast<DWORD>(request.offset & 0xFFFFFFFF);
    ctx->overlapped.OffsetHigh = static_cast<DWORD>(request.offset >> 32);
    
    pending_count_++;
    
    DWORD bytes_to_read = static_cast<DWORD>(std::min(request.size, 
        static_cast<uint64_t>(std::numeric_limits<DWORD>::max())));
    
    BOOL ok = ReadFile(file_handle_, request.buffer, bytes_to_read, 
                       nullptr, &ctx->overlapped);
    
    if (!ok && GetLastError() != ERROR_IO_PENDING) {
        // Sync completion or error
        DWORD bytes_read = 0;
        if (GetOverlappedResult(file_handle_, &ctx->overlapped, &bytes_read, FALSE)) {
            ctx->callback(request.offset, request.buffer, bytes_read, true);
        } else {
            ctx->callback(request.offset, request.buffer, 0, false);
        }
        pending_count_--;
        delete ctx;
    }
    // Else: async pending, will be completed by worker thread
}

void AsyncSSDStreamer::submit_batch(std::vector<ReadRequest>&& requests) {
    for (auto& req : requests) {
        submit_read(std::move(req));
    }
}

void AsyncSSDStreamer::flush() {
    while (pending_count_ > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

double AsyncSSDStreamer::throughput_mbps() const {
    static auto start = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start).count();
    if (elapsed == 0) return 0.0;
    return (total_bytes_read_ / 1048576.0) / elapsed;
}

void AsyncSSDStreamer::completion_worker() {
    SL_LOG_INFO("AsyncSSDStreamer: I/O completion worker started");
    
    while (running_) {
        DWORD bytes_transferred = 0;
        ULONG_PTR completion_key = 0;
        OVERLAPPED* overlapped = nullptr;
        
        BOOL ok = GetQueuedCompletionStatus(
            iocp_, 
            &bytes_transferred, 
            &completion_key, 
            &overlapped, 
            100  // 100ms timeout to check running_ flag
        );
        
        if (!overlapped) continue;  // Timeout or error
        
        auto* ctx = reinterpret_cast<OverlappedContext*>(
            reinterpret_cast<char*>(overlapped) - offsetof(OverlappedContext, overlapped));
        
        pending_count_--;
        total_bytes_read_ += bytes_transferred;
        
        if (ctx->callback) {
            ctx->callback(ctx->request_id, ctx->buffer, bytes_transferred, ok == TRUE);
        }
        
        delete ctx;
    }
    
    SL_LOG_INFO("AsyncSSDStreamer: I/O completion worker stopped");
}

} // namespace sl
