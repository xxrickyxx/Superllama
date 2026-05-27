#pragma once
#include "common/platform.h"


#include "tensor.h"
#include "common/types.h"

namespace sl {

// ====================================================================
// MLP / Feed-Forward Network block
// Supports LLaMA-style gate + up + down, and standard 2-layer MLP
// ====================================================================
class MLPBlock {
public:
    enum class MLPType {
        GATED_SILU,     // LLaMA/Mistral: gate + silu * up → down
        GELU,           // Older BERT-style
        SWIGLU,         // Alternative
        STANDARD        // Simple 2-layer
    };
    
    MLPBlock(const ModelConfig& config, uint32_t layer_idx, MLPType type = MLPType::GATED_SILU);
    
    // Forward pass
    // x: input [n_embd]  
    // output: [n_embd]
    void forward(const float* x, float* output);
    
    // Weight loading (pageable)
    void load_weights(const void* gate_w, const void* up_w, const void* down_w,
                      const void* fc1_w, const void* fc2_w);
    
    size_t total_weight_bytes() const;
    
private:
    uint32_t n_embd_;
    uint32_t n_ff_;         // Intermediate (hidden) dimension
    MLPType mlp_type_;
    
    // Weight tensors (SSD-pageable)
    Tensor gate_weight_;    // [n_embd, n_ff] - gate projection
    Tensor up_weight_;      // [n_embd, n_ff] - up projection  
    Tensor down_weight_;    // [n_ff, n_embd] - down projection
    
    // Scratch buffers
    std::vector<float> gate_buf_;   // [n_ff]
    std::vector<float> up_buf_;     // [n_ff]
    std::vector<float> hidden_buf_; // [n_ff]
};

} // namespace sl