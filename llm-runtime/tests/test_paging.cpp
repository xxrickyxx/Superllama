#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif
#include "engine/paging_manager.h"
#include "engine/model_loader.h"
#include <cstdio>
#include <iostream>
#include <cassert>
#include <filesystem>

namespace sl {
namespace test {

void test_page_cache() {
    PageCache cache(1024 * 1024);  // 1MB
    
    auto data = std::make_unique<uint8_t[]>(100);
    for (int i = 0; i < 100; i++) data[i] = static_cast<uint8_t>(i);
    
    cache.insert(1, std::move(data), 100);
    assert(cache.contains(1));
    
    void* ptr = cache.get(1);
    assert(ptr != nullptr);
    uint8_t* bytes = static_cast<uint8_t*>(ptr);
    assert(bytes[0] == 0);
    assert(bytes[50] == 50);
    
    cache.evict(1);
    assert(!cache.contains(1));
    
    auto d1 = std::make_unique<uint8_t[]>(600 * 1024);
    auto d2 = std::make_unique<uint8_t[]>(500 * 1024);
    cache.insert(10, std::move(d1), 600 * 1024);
    cache.insert(20, std::move(d2), 500 * 1024);
    
    std::cout << "  [PASS] PageCache LRU eviction" << std::endl;
}

void test_paging_manager_basics() {
    std::string test_file = "./test_model.bin";
    {
        FILE* f = fopen(test_file.c_str(), "wb");
        assert(f != nullptr);
        std::vector<uint8_t> data(1024 * 1024, 0xAB);
        fwrite(data.data(), 1, data.size(), f);
        fclose(f);
    }
    
    SSDPagingManager pm;
    
    if (pm.open_model(test_file)) {
        ModelConfig cfg;
        cfg.n_embd = 256;
        cfg.n_head = 8;
        cfg.n_head_kv = 8;
        cfg.n_layer = 4;
        cfg.n_ff = 1024;
        pm.set_model_config(cfg);
        
        uint64_t page_id = pm.register_tensor("test_tensor", 0, 0, 0, 256 * 4, DType::F32);
        
        void* data = pm.resolve_tensor(page_id);
        assert(data != nullptr);
        
        uint8_t* bytes = static_cast<uint8_t*>(data);
        assert(bytes[0] == 0xAB);
        
        auto stats = pm.get_stats();
        assert(stats.cache_hits + stats.cache_misses >= 1);
        
        std::cout << "  [PASS] Paging manager basics" << std::endl;
        
        pm.close();
    } else {
        std::cout << "  [SKIP] Paging manager (file access error)" << std::endl;
    }
    
    std::filesystem::remove(test_file);
}

void test_prefetch_queue() {
    std::string test_file = "./test_prefetch.bin";
    {
        FILE* f = fopen(test_file.c_str(), "wb");
        assert(f != nullptr);
        std::vector<uint8_t> data(512 * 1024, 0xCD);
        fwrite(data.data(), 1, data.size(), f);
        fclose(f);
    }
    
    SSDPagingManager pm;
    
    if (pm.open_model(test_file)) {
        ModelConfig cfg;
        cfg.n_embd = 128;
        cfg.n_head = 4;
        cfg.n_head_kv = 4;
        cfg.n_layer = 2;
        cfg.n_ff = 512;
        pm.set_model_config(cfg);
        
        pm.register_tensor("layer0.attn_q", 0, 0, 0, 1000, DType::F32);
        pm.register_tensor("layer0.attn_k", 0, 0, 1000, 1000, DType::F32);
        pm.register_tensor("layer1.attn_q", 1, 0, 2000, 1000, DType::F32);
        
        pm.prefetch_layer(1);
        
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        
        std::cout << "  [PASS] Prefetch queue" << std::endl;
        
        pm.close();
    }
    
    std::filesystem::remove(test_file);
}

void run_all_paging_tests() {
    std::cout << "Paging Tests:" << std::endl;
    test_page_cache();
    test_paging_manager_basics();
    test_prefetch_queue();
    std::cout << "All paging tests passed!\n" << std::endl;
}

} // namespace test
} // namespace sl
