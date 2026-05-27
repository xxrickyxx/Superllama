#pragma once
#include "common/platform.h"

#include "common/types.h"
#include "core/tensor.h"
#include <windows.h>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <list>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <queue>
#include <chrono>

namespace sl {

// ====================================================================
// PageDescriptor - metadata for a single page (tensor chunk on SSD)
// ====================================================================
struct PageDescriptor {
    uint64_t page_id;           // Unique page ID
    uint64_t file_offset;       // Offset within the model file on SSD
    uint64_t size_bytes;        // Size of this page
    uint32_t layer_idx;         // Which transformer layer (0..n_layer-1)
    uint32_t tensor_type;       // 0=attention, 1=mlp, 2=norm, 3=embed, 4=lm_head
    std::string tensor_name;    // e.g., "q_weight", "gate_weight"
    DType dtype;
    
    // Access pattern metadata for prefetch optimization
    uint32_t access_count;      // Times this page was accessed
    std::chrono::steady_clock::time_point last_access;
    bool is_active;             // Currently needed for inference
    bool is_prefetchable;       // Can be prefetched
};

// ====================================================================
// PageCache - LRU cache for resident pages
// ====================================================================
class PageCache {
public:
    static constexpr size_t DEFAULT_MAX_CACHE_MB = 4096;  // 4GB default cache
    
    explicit PageCache(size_t max_bytes = DEFAULT_MAX_CACHE_MB * 1024 * 1024);
    
    // Get a page into cache (may trigger eviction)
    // Returns pointer to resident memory, or nullptr if page not found
    void* get(uint64_t page_id);
    
    // Insert a page into cache
    void insert(uint64_t page_id, std::unique_ptr<uint8_t[]> data, size_t size);
    
    // Check if page is cached
    bool contains(uint64_t page_id) const;
    
    // Evict a specific page
    void evict(uint64_t page_id);
    
    // Evict LRU pages until total cache size is below limit
    void evict_to_fit(size_t needed_bytes);
    
    // Statistics
    size_t current_cache_bytes() const { return current_cache_bytes_; }
    size_t max_cache_bytes() const { return max_cache_bytes_; }
    double hit_rate() const;
    
    // Reset statistics
    void reset_stats();
    
private:
    struct CacheEntry {
        uint64_t page_id;
        std::unique_ptr<uint8_t[]> data;
        size_t size;
    };
    
    mutable std::mutex mutex_;
    std::unordered_map<uint64_t, std::list<CacheEntry>::iterator> cache_map_;
    std::list<CacheEntry> lru_list_;  // Front = MRU, Back = LRU
    size_t current_cache_bytes_ = 0;
    size_t max_cache_bytes_;
    
    mutable uint64_t total_hits_ = 0;
    mutable uint64_t total_misses_ = 0;
    
    void touch(std::list<CacheEntry>::iterator it);
    void remove(std::list<CacheEntry>::iterator it);
};

// ====================================================================
// SSDPagingManager - the core out-of-core memory system
// 
// Architecture:
// ┌─────────────────────────────────────────────────────────┐
// │                     CPU / GPU Compute                    │
// │  Active Layer Tensors (resolved via PageCache)          │
// ├─────────────────────────────────────────────────────────┤
// │               PageCache (LRU, 4GB default)              │
// │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐         │
// │  │ L0-Q │ │ L0-K │ │ L5-G │ │ L5-U │ │  ... │         │
// │  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘         │
// ├─────────────────────────────────────────────────────────┤
// │            Async Prefetch Queue (IOCP)                  │
// │  ┌──────┐ ┌──────┐ ┌──────┐                            │
// │  │ L6-Q │ │ L6-K │ │ L7-G │  (next layers predicted)   │
// │  └──────┘ └──────┘ └──────┘                            │
// ├─────────────────────────────────────────────────────────┤
// │      Memory-Mapped Model File (SSD / NVMe)             │
// │  Layer0 │ Layer1 │ Layer2 │ ... │ LayerN               │
// │  [Attn] [MLP] [Attn] [MLP] ... [Attn] [MLP]           │
// └─────────────────────────────────────────────────────────┘
// ====================================================================
class SSDPagingManager {
public:
    SSDPagingManager();
    ~SSDPagingManager();
    
    // Initialize with a model file
    // mmap_flags: 0=normal, 1=prefetch_sequential, 2=random_access
    bool open_model(const std::string& model_path);
    void close();
    
    // Register model structure
    void set_model_config(const ModelConfig& config);
    
    // Register a tensor as pageable (maps logical tensor to file region)
    uint64_t register_tensor(const std::string& name, 
                              uint32_t layer_idx,
                              uint32_t tensor_type,
                              uint64_t file_offset, 
                              uint64_t size_bytes,
                              DType dtype);
    
    // Get tensor data, ensuring it's resident in memory
    // Returns pointer to data (may block if page needs to be loaded)
    void* resolve_tensor(uint64_t page_id);
    
    // Schedule prefetch of tensor (non-blocking, async)
    // Used to pre-load next layers while current layer is computing
    void prefetch_tensor(uint64_t page_id, int priority = 0);
    
    // Prefetch entire layer (all tensors for a given layer)
    void prefetch_layer(uint32_t layer_idx);
    
    // Prefetch predicted next layers (lookahead)
    // lookahead_count: how many future layers to prefetch
    void prefetch_next_layers(uint32_t current_layer, uint32_t lookahead_count = 2);
    
    // Notify which layer is currently active (for prefetch scheduling)
    void set_active_layer(uint32_t layer_idx);
    
    // Evict layers that won't be needed soon
    void evict_distant_layers(uint32_t current_layer, uint32_t keep_radius = 2);
    
    // Statistics
    PagingStats get_stats() const;
    void reset_stats();
    
    // Cache configuration
    void set_max_cache_mb(size_t mb);
    size_t max_cache_mb() const;
    
    // Check if model fits entirely in available RAM
    bool model_fits_in_ram() const;
    
private:
    // Memory-mapped file handle
    HANDLE file_handle_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_handle_ = nullptr;
    void* mapped_base_ = nullptr;      // Base address of memory mapping
    uint64_t mapped_size_ = 0;
    std::string model_path_;
    
    // Page cache
    PageCache cache_;
    
    // Page descriptors
    std::unordered_map<uint64_t, PageDescriptor> page_descriptors_;
    std::unordered_map<std::string, uint64_t> tensor_name_to_page_;
    mutable std::mutex descriptor_mutex_;
    
    // Model structure
    ModelConfig config_;
    uint32_t n_layers_ = 0;
    
    // Prefetch queue (async I/O)
    struct PrefetchRequest {
        uint64_t page_id;
        int priority;
        std::chrono::steady_clock::time_point enqueue_time;
        
        bool operator<(const PrefetchRequest& other) const {
            return priority < other.priority;  // Higher priority first
        }
    };
    
    std::priority_queue<PrefetchRequest> prefetch_queue_;
    std::mutex prefetch_mutex_;
    std::condition_variable prefetch_cv_;
    std::atomic<bool> prefetch_running_{false};
    std::thread prefetch_thread_;
    
    // Overlapped I/O completion port
    HANDLE iocp_handle_ = nullptr;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    PagingStats stats_;
    std::atomic<uint32_t> active_layer_{0};
    
    // Internal methods
    void prefetch_worker();
    bool read_page_from_ssd(uint64_t page_id, void* buffer, uint64_t size);
    void* map_page_direct(uint64_t page_id);  // Direct mmap access (zero-copy)
    void complete_prefetch(uint64_t page_id);
};

// ====================================================================
// LayerTensorCache - caches all tensors for an active layer
// Holds resolved pointers during layer execution
// ====================================================================
class LayerTensorCache {
public:
    LayerTensorCache(SSDPagingManager& paging_mgr, uint32_t layer_idx);
    ~LayerTensorCache();
    
    // Resolve all tensors for a layer (blocks until all resident)
    void resolve_all();
    
    // Get raw pointer to tensor data
    void* get_attention_q() const { return attn_q_; }
    void* get_attention_k() const { return attn_k_; }
    void* get_attention_v() const { return attn_v_; }
    void* get_attention_o() const { return attn_o_; }
    void* get_mlp_gate() const { return mlp_gate_; }
    void* get_mlp_up() const { return mlp_up_; }
    void* get_mlp_down() const { return mlp_down_; }
    void* get_attn_norm() const { return attn_norm_; }
    void* get_ffn_norm() const { return ffn_norm_; }
    
    // Release all resolved pointers (allow eviction)
    void release_all();
    
    uint32_t layer_idx() const { return layer_idx_; }
    
private:
    SSDPagingManager& paging_mgr_;
    uint32_t layer_idx_;
    
    void* attn_q_ = nullptr;
    void* attn_k_ = nullptr;
    void* attn_v_ = nullptr;
    void* attn_o_ = nullptr;
    void* mlp_gate_ = nullptr;
    void* mlp_up_ = nullptr;
    void* mlp_down_ = nullptr;
    void* attn_norm_ = nullptr;
    void* ffn_norm_ = nullptr;
    
    bool resolved_ = false;
};

} // namespace sl