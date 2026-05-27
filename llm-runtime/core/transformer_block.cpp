#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif
#include "transformer_block.h"
#include "common/logging.h"
#include <cstring>

namespace sl {

TransformerBlock::TransformerBlock(const ModelConfig& config, uint32_t layer_idx)
    : n_embd_(config.n_embd)
    , layer_idx_(layer_idx)
    , attention_(config, layer_idx)
    , mlp_(config, layer_idx, MLPBlock::MLPType::GATED_SILU)
{
    attn_norm_weight_ = Tensor({n_embd_}, DType::F32);
    ffn_norm_weight_  = Tensor({n_embd_}, DType::F32);
    
    residual_buf_.resize(n_embd_);
    norm_buf_.resize(n_embd_);
    attn_out_buf_.resize(n_embd_);
    mlp_out_buf_.resize(n_embd_);
}

void TransformerBlock::set_attention_weights(const void* q, const void* k, 
                                              const void* v, const void* o) {
    attention_.load_weights(q, k, v, o, nullptr, nullptr, nullptr, nullptr);
}

void TransformerBlock::set_mlp_weights(const void* gate, const void* up, const void* down) {
    mlp_.load_weights(gate, up, down, nullptr, nullptr);
}

void TransformerBlock::set_norm_weights(const void* attn_norm, const void* ffn_norm) {
    memcpy(attn_norm_weight_.data(), attn_norm, n_embd_ * sizeof(float));
    memcpy(ffn_norm_weight_.data(), ffn_norm, n_embd_ * sizeof(float));
}

void TransformerBlock::rms_norm(const float* input, const float* weight, float* output) {
    // First compute RMS and normalize
    simd::rms_norm_f32(input, output, n_embd_, 1e-6f);
    // Apply learnable scale (weight)
    for (uint32_t i = 0; i + 8 <= n_embd_; i += 8) {
        __m256 v = _mm256_loadu_ps(output + i);
        __m256 w = _mm256_loadu_ps(weight + i);
        _mm256_storeu_ps(output + i, _mm256_mul_ps(v, w));
    }
}

void TransformerBlock::forward(float* hidden, KVCacheSlot* kv_cache, uint32_t position) {
    // Save residual
    memcpy(residual_buf_.data(), hidden, n_embd_ * sizeof(float));
    
    // Pre-attention RMSNorm
    rms_norm(hidden, static_cast<const float*>(attn_norm_weight_.data()), norm_buf_.data());
    
    // Attention
    attention_.forward(norm_buf_.data(), kv_cache, position, attn_out_buf_.data());
    
    // Residual connection
    for (uint32_t i = 0; i + 8 <= n_embd_; i += 8) {
        __m256 r = _mm256_loadu_ps(residual_buf_.data() + i);
        __m256 a = _mm256_loadu_ps(attn_out_buf_.data() + i);
        _mm256_storeu_ps(hidden + i, _mm256_add_ps(r, a));
    }
    
    // Save residual again
    memcpy(residual_buf_.data(), hidden, n_embd_ * sizeof(float));
    
    // Pre-FFN RMSNorm
    rms_norm(hidden, static_cast<const float*>(ffn_norm_weight_.data()), norm_buf_.data());
    
    // MLP
    mlp_.forward(norm_buf_.data(), mlp_out_buf_.data());
    
    // Second residual
    for (uint32_t i = 0; i + 8 <= n_embd_; i += 8) {
        __m256 r = _mm256_loadu_ps(residual_buf_.data() + i);
        __m256 m = _mm256_loadu_ps(mlp_out_buf_.data() + i);
        _mm256_storeu_ps(hidden + i, _mm256_add_ps(r, m));
    }
}

size_t TransformerBlock::total_weight_bytes() const {
    return mlp_.total_weight_bytes() 
         + attn_norm_weight_.nbytes() 
         + ffn_norm_weight_.nbytes();
}

} // namespace sl