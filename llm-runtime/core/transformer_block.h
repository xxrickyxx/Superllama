#pragma once
#include "common/platform.h"


#include "attention.h"
#include "mlp.h"
#include "common/types.h"
#include <memory>

namespace sl {

// ====================================================================
// Single Transformer Block (decoder layer)
// LLaMA-style: RMSNorm → Attention → Residual → RMSNorm → MLP → Residual
// ====================================================================
class TransformerBlock {
public:
    TransformerBlock(const ModelConfig& config, uint32_t layer_idx);
    
    // Forward pass with KV cache
    // hidden: [n_embd] input/output in-place
    // kv_cache: pointer to KV cache slots for this layer
    // position: current sequence position
    void forward(float* hidden, KVCacheSlot* kv_cache, uint32_t position);
    
    // Set weight data (pageable pointers)
    void set_attention_weights(const void* q, const void* k, const void* v, const void* o);
    void set_mlp_weights(const void* gate, const void* up, const void* down);
    void set_norm_weights(const void* attn_norm, const void* ffn_norm);
    
    size_t total_weight_bytes() const;
    
private:
    uint32_t n_embd_;
    uint32_t layer_idx_;
    
    // Sub-modules
    MultiHeadAttention attention_;
    MLPBlock mlp_;
    
    // RMS Norm weights
    Tensor attn_norm_weight_;  // [n_embd]
    Tensor ffn_norm_weight_;   // [n_embd]
    
    // Scratch buffers
    std::vector<float> residual_buf_;
    std::vector<float> norm_buf_;
    std::vector<float> attn_out_buf_;
    std::vector<float> mlp_out_buf_;
    
    void rms_norm(const float* input, const float* weight, float* output);
};

} // namespace sl