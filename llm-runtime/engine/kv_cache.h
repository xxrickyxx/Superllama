#pragma once
#include "common/platform.h"

#include "common/types.h"
#include <vector>
#include <memory>
#include <string>
#include <utility>
#include <cstdint>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace sl {

class KVCacheManager {
public:
    KVCacheManager(uint32_t n_layers, uint32_t n_kv_head, 
                   uint32_t head_dim, uint32_t max_seq_len);
    ~KVCacheManager();
    
    KVCacheSlot* get_layer_cache(uint32_t layer_idx);
    const KVCacheSlot* get_layer_cache(uint32_t layer_idx) const;
    
    void store(uint32_t layer_idx, uint32_t position, 
               const float* k, const float* v);
    
    std::pair<const float*, const float*> retrieve(uint32_t layer_idx, 
                                                    uint32_t position) const;
    
    void clear();
    void truncate(uint32_t new_length);
    
    size_t total_allocated_bytes() const;
    size_t used_entries() const { return current_length_; }
    size_t max_entries() const { return max_seq_len_; }
    
    bool evict_to_ssd(uint32_t layer_idx, uint32_t start_pos, uint32_t end_pos);
    bool restore_from_ssd(uint32_t layer_idx, uint32_t start_pos, uint32_t end_pos);
    
private:
    uint32_t n_layers_;
    uint32_t n_kv_head_;
    uint32_t head_dim_;
    uint32_t max_seq_len_;
    uint32_t current_length_ = 0;
    
    std::vector<std::vector<KVCacheSlot>> layer_caches_;
    size_t slot_size_bytes_ = 0;
};

#ifdef _WIN32
class KVCachePager {
public:
    KVCachePager(const std::string& ssd_cache_path, size_t max_ram_mb = 2048);
    ~KVCachePager();
    
    void attach(KVCacheManager* cache);
    void evict_cold_entries(uint32_t keep_radius = 64);
    void prefetch_window(uint32_t center_pos, uint32_t window_size = 128);
    
private:
    struct PagedEntry {
        uint32_t layer_idx;
        uint32_t position;
        std::vector<uint8_t> serialized_kv;
    };
    
    std::string ssd_path_;
    size_t max_ram_mb_;
    KVCacheManager* attached_cache_ = nullptr;
    
    HANDLE cache_file_handle_ = INVALID_HANDLE_VALUE;
    std::vector<PagedEntry> paged_entries_;
};
#endif

} // namespace sl


