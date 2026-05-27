#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif
#include "kv_cache.h"
#include "common/logging.h"
#include <cstring>

namespace sl {

KVCacheManager::KVCacheManager(uint32_t n_layers, uint32_t n_kv_head,
                                uint32_t head_dim, uint32_t max_seq_len)
    : n_layers_(n_layers)
    , n_kv_head_(n_kv_head)
    , head_dim_(head_dim)
    , max_seq_len_(max_seq_len)
{
    slot_size_bytes_ = n_kv_head_ * head_dim_ * sizeof(float);
    
    // Pre-allocate all KV cache slots (this is the big allocation)
    layer_caches_.resize(n_layers_);
    for (auto& layer : layer_caches_) {
        layer.resize(max_seq_len_);
        for (auto& slot : layer) {
            slot.k.resize(n_kv_head_ * head_dim_);
            slot.v.resize(n_kv_head_ * head_dim_);
        }
    }
    
    size_t total = total_allocated_bytes();
    SL_LOG_INFO("KVCacheManager: allocated {:.1f} MB for {} layers x {} seq",
                total / 1048576.0, n_layers_, max_seq_len_);
}

KVCacheManager::~KVCacheManager() = default;

KVCacheSlot* KVCacheManager::get_layer_cache(uint32_t layer_idx) {
    if (layer_idx >= n_layers_) return nullptr;
    return layer_caches_[layer_idx].data();
}

const KVCacheSlot* KVCacheManager::get_layer_cache(uint32_t layer_idx) const {
    if (layer_idx >= n_layers_) return nullptr;
    return layer_caches_[layer_idx].data();
}

void KVCacheManager::store(uint32_t layer_idx, uint32_t position,
                            const float* k, const float* v) {
    if (layer_idx >= n_layers_ || position >= max_seq_len_) return;
    
    auto& slot = layer_caches_[layer_idx][position];
    size_t size = n_kv_head_ * head_dim_;
    memcpy(slot.k.data(), k, size * sizeof(float));
    memcpy(slot.v.data(), v, size * sizeof(float));
    
    if (position + 1 > current_length_) {
        current_length_ = position + 1;
    }
}

std::pair<const float*, const float*> KVCacheManager::retrieve(
    uint32_t layer_idx, uint32_t position) const {
    if (layer_idx >= n_layers_ || position >= max_seq_len_) {
        return {nullptr, nullptr};
    }
    const auto& slot = layer_caches_[layer_idx][position];
    return {slot.k.data(), slot.v.data()};
}

void KVCacheManager::clear() {
    current_length_ = 0;
    for (auto& layer : layer_caches_) {
        for (auto& slot : layer) {
            std::fill(slot.k.begin(), slot.k.end(), 0.0f);
            std::fill(slot.v.begin(), slot.v.end(), 0.0f);
        }
    }
}

void KVCacheManager::truncate(uint32_t new_length) {
    current_length_ = std::min(new_length, max_seq_len_);
}

size_t KVCacheManager::total_allocated_bytes() const {
    // K and V for all layers x positions
    return n_layers_ * max_seq_len_ * 2 * n_kv_head_ * head_dim_ * sizeof(float);
}

bool KVCacheManager::evict_to_ssd(uint32_t layer_idx, uint32_t start_pos, uint32_t end_pos) {
    // Simplified: in production would serialize and write to swap file
    SL_LOG_DEBUG("KVCacheManager: evicting layer {} positions [{}, {})", 
                 layer_idx, start_pos, end_pos);
    return true;
}

bool KVCacheManager::restore_from_ssd(uint32_t layer_idx, uint32_t start_pos, uint32_t end_pos) {
    SL_LOG_DEBUG("KVCacheManager: restoring layer {} positions [{}, {})",
                 layer_idx, start_pos, end_pos);
    return true;
}

// ====================================================================
// KVCachePager (simplified stub for production extension)
// ====================================================================
KVCachePager::KVCachePager(const std::string& ssd_cache_path, size_t max_ram_mb)
    : ssd_path_(ssd_cache_path)
    , max_ram_mb_(max_ram_mb) {}

KVCachePager::~KVCachePager() {
    if (cache_file_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(cache_file_handle_);
    }
}

void KVCachePager::attach(KVCacheManager* cache) {
    attached_cache_ = cache;
}

void KVCachePager::evict_cold_entries(uint32_t keep_radius) {
    if (!attached_cache_) return;
    SL_LOG_TRACE("KVCachePager: evicting cold entries (keep_radius={})", keep_radius);
}

void KVCachePager::prefetch_window(uint32_t center_pos, uint32_t window_size) {
    SL_LOG_TRACE("KVCachePager: prefetching window around pos {}", center_pos);
}

} // namespace sl
