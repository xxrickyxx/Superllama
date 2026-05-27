#include "paging_manager.h"
#include "common/logging.h"
#include <algorithm>
#include <cassert>

namespace sl {

// ====================================================================
// PageCache implementation
// ====================================================================
PageCache::PageCache(size_t max_bytes)
    : max_cache_bytes_(max_bytes) {}

void* PageCache::get(uint64_t page_id) {
    std::lock_guard lock(mutex_);
    auto it = cache_map_.find(page_id);
    if (it != cache_map_.end()) {
        total_hits_++;
        touch(it->second);  // Move to front (MRU)
        return it->second->data.get();
    }
    total_misses_++;
    return nullptr;
}

void PageCache::insert(uint64_t page_id, std::unique_ptr<uint8_t[]> data, size_t size) {
    if (!data) return;
    
    std::lock_guard lock(mutex_);
    
    // Ensure space
    if (current_cache_bytes_ + size > max_cache_bytes_) {
        evict_to_fit(size);
    }
    
    // Create entry
    CacheEntry entry{page_id, std::move(data), size};
    lru_list_.push_front(std::move(entry));
    cache_map_[page_id] = lru_list_.begin();
    current_cache_bytes_ += size;
}

bool PageCache::contains(uint64_t page_id) const {
    std::lock_guard lock(mutex_);
    return cache_map_.find(page_id) != cache_map_.end();
}

void PageCache::evict(uint64_t page_id) {
    std::lock_guard lock(mutex_);
    auto it = cache_map_.find(page_id);
    if (it != cache_map_.end()) {
        remove(it->second);
        cache_map_.erase(it);
    }
}

void PageCache::evict_to_fit(size_t needed_bytes) {
    // Evict from LRU (back of list) until we have space
    while (current_cache_bytes_ + needed_bytes > max_cache_bytes_ && !lru_list_.empty()) {
        auto& entry = lru_list_.back();
        SL_LOG_DEBUG("PageCache: evicting page {} ({} bytes, LRU)", 
                      entry.page_id, entry.size);
        remove(--lru_list_.end());  // Remove last (LRU)
        cache_map_.erase(entry.page_id);
    }
}

double PageCache::hit_rate() const {
    std::lock_guard lock(mutex_);
    uint64_t total = total_hits_ + total_misses_;
    return total > 0 ? static_cast<double>(total_hits_) / total : 0.0;
}

void PageCache::reset_stats() {
    std::lock_guard lock(mutex_);
    total_hits_ = 0;
    total_misses_ = 0;
}

void PageCache::touch(std::list<CacheEntry>::iterator it) {
    // Move to front (MRU position)
    lru_list_.splice(lru_list_.begin(), lru_list_, it);
}

void PageCache::remove(std::list<CacheEntry>::iterator it) {
    current_cache_bytes_ -= it->size;
    lru_list_.erase(it);
}

// ====================================================================
// SSDPagingManager implementation
// ====================================================================
SSDPagingManager::SSDPagingManager()
    : cache_(PageCache::DEFAULT_MAX_CACHE_MB * 1024 * 1024) {
    // Create IOCP for async I/O
    iocp_handle_ = CreateIoCompletionPort(INVALID_HANDLE_VALUE, nullptr, 0, 0);
    
    // Start prefetch worker thread
    prefetch_running_ = true;
    prefetch_thread_ = std::thread(&SSDPagingManager::prefetch_worker, this);
}

SSDPagingManager::~SSDPagingManager() {
    prefetch_running_ = false;
    prefetch_cv_.notify_all();
    if (prefetch_thread_.joinable()) {
        prefetch_thread_.join();
    }
    close();
    if (iocp_handle_) {
        CloseHandle(iocp_handle_);
    }
}

bool SSDPagingManager::open_model(const std::string& model_path) {
    close();
    
    // Open file
    file_handle_ = CreateFileA(
        model_path.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN,  // Optimize for sequential SSD reads
        nullptr
    );
    
    if (file_handle_ == INVALID_HANDLE_VALUE) {
        SL_LOG_ERROR("Failed to open model file: {}", model_path);
        return false;
    }
    
    // Get file size
    LARGE_INTEGER li;
    GetFileSizeEx(file_handle_, &li);
    mapped_size_ = li.QuadPart;
    
    // Create file mapping (memory-mapped file for zero-copy access)
    mapping_handle_ = CreateFileMappingA(
        file_handle_,
        nullptr,
        PAGE_READONLY,
        0, 0,
        nullptr
    );
    
    if (!mapping_handle_) {
        SL_LOG_ERROR("Failed to create file mapping");
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
        return false;
    }
    
    // Map view (partial mapping - we'll map chunks on demand)
    // Initially, we just reserve address space; pages are faulted in by OS
    mapped_base_ = MapViewOfFile(
        mapping_handle_,
        FILE_MAP_READ,
        0, 0,
        0  // Map entire file
    );
    
    if (!mapped_base_) {
        // Fallback: map a smaller initial chunk
        SL_LOG_WARN("Full file map failed, using chunked mapping");
        mapped_base_ = MapViewOfFile(
            mapping_handle_,
            FILE_MAP_READ,
            0, 0,
            std::min<uint64_t>(mapped_size_, 256ULL * 1024 * 1024) // 256MB initial
        );
    }
    
    model_path_ = model_path;
    
    // Associate file handle with IOCP
    CreateIoCompletionPort(file_handle_, iocp_handle_, 0, 0);
    
    SL_LOG_INFO("SSDPagingManager: opened model '{}' ({:.1f} GB)", 
                 model_path, mapped_size_ / (1024.0 * 1024.0 * 1024.0));
    
    stats_.total_model_bytes = mapped_size_;
    stats_.mapped_bytes = mapped_size_;
    
    return true;
}

void SSDPagingManager::close() {
    if (mapped_base_) {
        UnmapViewOfFile(mapped_base_);
        mapped_base_ = nullptr;
    }
    if (mapping_handle_) {
        CloseHandle(mapping_handle_);
        mapping_handle_ = nullptr;
    }
    if (file_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
    }
    mapped_size_ = 0;
}

void SSDPagingManager::set_model_config(const ModelConfig& config) {
    config_ = config;
    n_layers_ = config.n_layer;
}

uint64_t SSDPagingManager::register_tensor(
    const std::string& name,
    uint32_t layer_idx,
    uint32_t tensor_type,
    uint64_t file_offset,
    uint64_t size_bytes,
    DType dtype)
{
    std::lock_guard lock(descriptor_mutex_);
    
    static std::atomic<uint64_t> next_page_id{1};
    uint64_t page_id = next_page_id++;
    
    PageDescriptor desc{
        .page_id = page_id,
        .file_offset = file_offset,
        .size_bytes = size_bytes,
        .layer_idx = layer_idx,
        .tensor_type = tensor_type,
        .tensor_name = name,
        .dtype = dtype,
        .access_count = 0,
        .last_access = std::chrono::steady_clock::now(),
        .is_active = false,
        .is_prefetchable = true
    };
    
    page_descriptors_[page_id] = desc;
    tensor_name_to_page_[name] = page_id;
    
    return page_id;
}

void* SSDPagingManager::resolve_tensor(uint64_t page_id) {
    // First check cache
    void* cached = cache_.get(page_id);
    if (cached) {
        stats_.cache_hits++;
        return cached;
    }
    
    stats_.cache_misses++;
    
    // Need to load from SSD
    PageDescriptor desc;
    {
        std::lock_guard<std::mutex> lock(descriptor_mutex_);
        auto it = page_descriptors_.find(page_id);
        if (it == page_descriptors_.end()) {
            SL_LOG_ERROR("Unknown page ID: {}", page_id);
            return nullptr;
        }
        desc = it->second;
    }
    
    // Try direct memory-mapped access first (fast path, zero-copy)
    if (mapped_base_ && desc.file_offset + desc.size_bytes <= mapped_size_) {
        void* ptr = static_cast<uint8_t*>(mapped_base_) + desc.file_offset;
        
        // Touch the pages to ensure they're faulted in
        // Prefetch to CPU cache
        volatile char touch = static_cast<volatile char*>(ptr)[0];
        (void)touch;
        
        // Create a cache entry copy for LRU tracking
        auto cached_data = std::make_unique<uint8_t[]>(desc.size_bytes);
        memcpy(cached_data.get(), ptr, desc.size_bytes);
        cache_.insert(page_id, std::move(cached_data), desc.size_bytes);
        
        stats_.resident_bytes += desc.size_bytes;
        stats_.io_bytes_read += desc.size_bytes;
        
        // Update access metadata
        {
            std::lock_guard<std::mutex> lock(descriptor_mutex_);
            auto& d = page_descriptors_[page_id];
            d.access_count++;
            d.last_access = std::chrono::steady_clock::now();
            d.is_active = true;
        }
        
        return cache_.get(page_id);
    }
    
    // Slow path: explicit read via overlapped I/O
    auto buffer = std::make_unique<uint8_t[]>(desc.size_bytes);
    
    if (read_page_from_ssd(page_id, buffer.get(), desc.size_bytes)) {
        cache_.insert(page_id, std::move(buffer), desc.size_bytes);
        stats_.resident_bytes += desc.size_bytes;
        stats_.io_bytes_read += desc.size_bytes;
        return cache_.get(page_id);
    }
    
    return nullptr;
}

void SSDPagingManager::prefetch_tensor(uint64_t page_id, int priority) {
    if (cache_.contains(page_id)) return;  // Already resident
    
    std::lock_guard<std::mutex> lock(prefetch_mutex_);
    prefetch_queue_.push({
        page_id, 
        priority, 
        std::chrono::steady_clock::now()
    });
    prefetch_cv_.notify_one();
    
    SL_LOG_TRACE("SSDPagingManager: enqueued prefetch for page {}", page_id);
}

void SSDPagingManager::prefetch_layer(uint32_t layer_idx) {
    // Prefetch all tensors for a given layer
    std::lock_guard lock(descriptor_mutex_);
    for (const auto& [page_id, desc] : page_descriptors_) {
        if (desc.layer_idx == layer_idx) {
            prefetch_tensor(page_id, 10);  // High priority for current layer
        }
    }
}

void SSDPagingManager::prefetch_next_layers(uint32_t current_layer, uint32_t lookahead_count) {
    // Predict and prefetch upcoming layers
    for (uint32_t offset = 1; offset <= lookahead_count; offset++) {
        uint32_t next_layer = current_layer + offset;
        if (next_layer < n_layers_) {
            prefetch_layer(next_layer);
        }
    }
}

void SSDPagingManager::set_active_layer(uint32_t layer_idx) {
    active_layer_ = layer_idx;
    
    // Trigger prefetch of upcoming layers
    prefetch_next_layers(layer_idx, 2);
    
    // Evict layers far in the past (won't be needed again soon)
    if (layer_idx > 2) {
        evict_distant_layers(layer_idx, 2);
    }
}

void SSDPagingManager::evict_distant_layers(uint32_t current_layer, uint32_t keep_radius) {
    std::lock_guard lock(descriptor_mutex_);
    for (auto& [page_id, desc] : page_descriptors_) {
        if (desc.layer_idx < current_layer - keep_radius ||
            desc.layer_idx > current_layer + keep_radius + 2) {
            if (desc.is_active) {
                cache_.evict(page_id);
                desc.is_active = false;
            }
        }
    }
}

PagingStats SSDPagingManager::get_stats() const {
    PagingStats s = stats_;
    s.cache_hit_rate = cache_.hit_rate();
    s.resident_bytes = cache_.current_cache_bytes();
    s.active_layers = active_layer_;
    s.total_layers = n_layers_;
    return s;
}

void SSDPagingManager::reset_stats() {
    cache_.reset_stats();
    stats_ = PagingStats{};
    stats_.total_model_bytes = mapped_size_;
    stats_.mapped_bytes = mapped_size_;
}

void SSDPagingManager::set_max_cache_mb(size_t mb) {
    // Cache is initialized in constructor; for now, dynamic resize is limited
    SL_LOG_INFO("Cache size configuration: {} MB", mb);
}

size_t SSDPagingManager::max_cache_mb() const {
    return cache_.max_cache_bytes() / (1024 * 1024);
}

bool SSDPagingManager::model_fits_in_ram() const {
    MEMORYSTATUSEX mem_status;
    mem_status.dwLength = sizeof(mem_status);
    GlobalMemoryStatusEx(&mem_status);
    return mapped_size_ < mem_status.ullAvailPhys;
}

void SSDPagingManager::prefetch_worker() {
    SL_LOG_INFO("SSDPagingManager: prefetch worker started");
    
    while (prefetch_running_) {
        PrefetchRequest req;
        {
            std::unique_lock<std::mutex> lock(prefetch_mutex_);
            prefetch_cv_.wait(lock, [this] {
                return !prefetch_queue_.empty() || !prefetch_running_;
            });
            
            if (!prefetch_running_) break;
            
            if (!prefetch_queue_.empty()) {
                req = prefetch_queue_.top();
                prefetch_queue_.pop();
            } else {
                continue;
            }
        }
        
        // Skip if already cached (race condition safe)
        if (cache_.contains(req.page_id)) {
            stats_.prefetch_hits++;
            continue;
        }
        
        // Load the page
        resolve_tensor(req.page_id);
        
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - req.enqueue_time).count();
        
        SL_LOG_TRACE("SSDPagingManager: prefetched page {} in {}ms", req.page_id, elapsed);
    }
    
    SL_LOG_INFO("SSDPagingManager: prefetch worker stopped");
}

bool SSDPagingManager::read_page_from_ssd(uint64_t page_id, void* buffer, uint64_t size) {
    PageDescriptor desc;
    {
        std::lock_guard lock(descriptor_mutex_);
        auto it = page_descriptors_.find(page_id);
        if (it == page_descriptors_.end()) return false;
        desc = it->second;
    }
    
    // Direct mapped access is preferred
    if (mapped_base_) {
        void* mapped_ptr = static_cast<uint8_t*>(mapped_base_) + desc.file_offset;
        if (desc.file_offset + size <= mapped_size_) {
            memcpy(buffer, mapped_ptr, size);
            return true;
        }
    }
    
    // Fallback: explicit ReadFile with OVERLAPPED for async
    OVERLAPPED overlapped = {};
    overlapped.Offset = static_cast<DWORD>(desc.file_offset & 0xFFFFFFFF);
    overlapped.OffsetHigh = static_cast<DWORD>(desc.file_offset >> 32);
    
    DWORD bytes_read = 0;
    BOOL ok = ReadFile(file_handle_, buffer, static_cast<DWORD>(size), &bytes_read, &overlapped);
    
    if (!ok && GetLastError() != ERROR_IO_PENDING) {
        SL_LOG_ERROR("ReadFile failed for page {}: error {}", page_id, GetLastError());
        return false;
    }
    
    // Wait for completion
    if (!ok) {
        DWORD transferred;
        if (!GetOverlappedResult(file_handle_, &overlapped, &transferred, TRUE)) {
            SL_LOG_ERROR("GetOverlappedResult failed: error {}", GetLastError());
            return false;
        }
    }
    
    return true;
}

void* SSDPagingManager::map_page_direct(uint64_t page_id) {
    // Direct zero-copy access via memory mapping
    if (!mapped_base_) return nullptr;
    
    PageDescriptor desc;
    {
        std::lock_guard lock(descriptor_mutex_);
        auto it = page_descriptors_.find(page_id);
        if (it == page_descriptors_.end()) return nullptr;
        desc = it->second;
    }
    
    return static_cast<uint8_t*>(mapped_base_) + desc.file_offset;
}

void SSDPagingManager::complete_prefetch(uint64_t page_id) {
    // Called when an async prefetch completes
    stats_.prefetch_hits++;
}

// ====================================================================
// LayerTensorCache implementation
// ====================================================================
LayerTensorCache::LayerTensorCache(SSDPagingManager& paging_mgr, uint32_t layer_idx)
    : paging_mgr_(paging_mgr)
    , layer_idx_(layer_idx) {}

LayerTensorCache::~LayerTensorCache() {
    release_all();
}

void LayerTensorCache::resolve_all() {
    if (resolved_) return;
    
    // Resolve each tensor by name pattern
    // In production, these page IDs would be obtained from model_loader
    // For now, we illustrate the pattern
    
    // This would be wired up via paging_mgr_.tensor_name_to_page_
    // e.g.:
    // attn_q_ = paging_mgr_.resolve_tensor(page_ids_["q_weight"]);
    
    resolved_ = true;
}

void LayerTensorCache::release_all() {
    // In a real system, we'd mark these as evictable
    attn_q_ = nullptr;
    attn_k_ = nullptr;
    attn_v_ = nullptr;
    attn_o_ = nullptr;
    mlp_gate_ = nullptr;
    mlp_up_ = nullptr;
    mlp_down_ = nullptr;
    attn_norm_ = nullptr;
    ffn_norm_ = nullptr;
    resolved_ = false;
}

} // namespace sl
