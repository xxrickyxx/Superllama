#pragma once
#include "common/platform.h"

#include "common/types.h"
#include "model_loader.h"
#include "paging_manager.h"
#include "kv_cache.h"
#include "core/tensor.h"
#include "core/transformer_block.h"
#include <atomic>
#include <thread>
#include <queue>
#include <condition_variable>
#include <functional>

namespace sl {

// ====================================================================
// InferenceEngine - top-level inference orchestrator
// 
// Coordinates:
// 1. Token-by-token autoregressive generation
// 2. Layer execution with SSD paging
// 3. KV cache management
// 4. Streaming output
// ====================================================================
class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine();
    
    // Initialize with a loaded model
    bool initialize(ModelLoader& loader);
    
    // Generate text
    // Returns immediately if streaming, calls callback for each token
    // Blocks until complete if !stream
    std::vector<InferenceToken> generate(
        const InferenceRequest& request,
        TokenCallback callback = nullptr);
    
    // Cancel ongoing generation
    void cancel();
    
    // Check if generation is in progress
    bool is_generating() const { return generating_; }
    
    // Statistics
    PagingStats get_paging_stats() const;
    
    // Warmup: preload frequently-used layers into cache
    void warmup();
    
private:
    ModelLoader* loader_ = nullptr;
    SSDPagingManager* paging_mgr_ = nullptr;
    ModelConfig config_;
    
    // KV cache per layer
    std::unique_ptr<KVCacheManager> kv_cache_;
    
    // Transformer blocks (compute objects, lightweight without weights)
    std::vector<std::unique_ptr<TransformerBlock>> blocks_;
    
    // Generation state
    std::atomic<bool> generating_{false};
    std::atomic<bool> cancel_requested_{false};
    
    // Token buffer
    std::vector<float> hidden_state_;   // [n_embd]
    std::vector<float> logits_;         // [n_vocab]
    std::vector<Token> generated_tokens_;
    
    // Embedding table (small, always resident, SSD-pageable though)
    Tensor token_embeddings_;           // [n_vocab, n_embd]
    Tensor lm_head_;                    // [n_embd, n_vocab] (often shared with embeddings)
    Tensor final_norm_;                 // [n_embd]
    
    // Tokenizer (placeholder - would be SentencePiece or BPE in production)
    struct TokenizerState {
        std::string vocab_path;
        std::vector<std::string> vocab;
        std::unordered_map<std::string, uint32_t> token_to_id;
    };
    TokenizerState tokenizer_;
    
    // Internal methods
    void load_embedding_weights(ModelLoader& loader);
    void create_transformer_blocks();
    void run_transformer_layer(uint32_t layer_idx);
    void compute_logits();
    Token sample_token(const GenerationConfig& gen_cfg);
    
    // Tokenizer (simplified)
    std::vector<Token> tokenize(const std::string& text);
    std::string detokenize(Token token_id);
    
    // Streaming output thread
    std::thread generation_thread_;
};

// ====================================================================
// SSDBackend - abstract interface for SSD I/O backends
// ====================================================================
class SSDBackend {
public:
    virtual ~SSDBackend() = default;
    virtual bool open(const std::string& path) = 0;
    virtual void close() = 0;
    virtual void read(uint64_t offset, void* buffer, uint64_t size) = 0;
    virtual void async_read(uint64_t offset, void* buffer, uint64_t size,
                            std::function<void(bool)> callback) = 0;
    virtual void* map(uint64_t offset, uint64_t size) = 0;
    virtual uint64_t file_size() const = 0;
};

// Windows-specific NVMe-optimized backend
class NVMeSSDBackend : public SSDBackend {
public:
    bool open(const std::string& path) override;
    void close() override;
    void read(uint64_t offset, void* buffer, uint64_t size) override;
    void async_read(uint64_t offset, void* buffer, uint64_t size,
                    std::function<void(bool)> callback) override;
    void* map(uint64_t offset, uint64_t size) override;
    uint64_t file_size() const override;
    
private:
    HANDLE file_handle_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_handle_ = nullptr;
    void* mapped_base_ = nullptr;
    uint64_t file_size_ = 0;
    HANDLE iocp_ = nullptr;
};

} // namespace sl