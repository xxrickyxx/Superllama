#pragma once
#include "common/platform.h"


#include "tensor.h"
#include "common/types.h"

namespace sl {

// ====================================================================
// Multi-Head Attention (with GQA / MQA support)
// Supports KV-cache for autoregressive inference
// ====================================================================
class MultiHeadAttention {
public:
    MultiHeadAttention(const ModelConfig& config, uint32_t layer_idx);
    
    // Forward pass with KV cache
    // x: input [n_embd]
    // kv_cache: persistent KV cache for this layer
    // position: current position in sequence
    // output: [n_embd]
    void forward(const float* x, KVCacheSlot* kv_cache, 
                 uint32_t position, float* output);
    
    // Weight loading
    void load_weights(const void* q_weight, const void* k_weight,
                      const void* v_weight, const void* o_weight,
                      const void* q_bias, const void* k_bias,
                      const void* v_bias, const void* o_bias);
    
    uint32_t n_head() const { return n_head_; }
    uint32_t n_kv_head() const { return n_kv_head_; }
    uint32_t head_dim() const { return head_dim_; }
    
private:
    uint32_t n_head_;
    uint32_t n_kv_head_;
    uint32_t head_dim_;
    uint32_t n_embd_;
    uint32_t layer_idx_;
    
    // Weight tensors (SSD-pageable)
    Tensor q_weight_;   // [n_embd, n_head * head_dim]
    Tensor k_weight_;   // [n_embd, n_kv_head * head_dim]
    Tensor v_weight_;   // [n_embd, n_kv_head * head_dim]
    Tensor o_weight_;   // [n_head * head_dim, n_embd]
    
    // Temporary buffers
    std::vector<float> q_buf_;   // [n_head * head_dim]
    std::vector<float> k_buf_;   // [n_kv_head * head_dim]
    std::vector<float> v_buf_;   // [n_kv_head * head_dim]
    std::vector<float> attn_buf_;// [n_head]
    std::vector<float> o_buf_;   // [n_embd]
    
    // MAS (Multi-head Attention Scaling)
    std::vector<float> scale_buf_;
    
    void compute_qkv(const float* x);
    void apply_rope(uint32_t position);
    void compute_attention_scores(uint32_t position, KVCacheSlot* kv_cache);
    void compute_output(float* output);
};

// ====================================================================
// Flash Attention - memory-efficient attention
// For large sequences, avoids materializing the full attention matrix
// ====================================================================
class FlashAttention {
public:
    FlashAttention(uint32_t head_dim, float scale);
    
    // Flash attention forward (no KV cache, for prefill)
    void forward(const float* q, const float* k, const float* v,
                 uint32_t seq_len, uint32_t n_head, float* output);
    
private:
    uint32_t head_dim_;
    float scale_;
};

} // namespace sl