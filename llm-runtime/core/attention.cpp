#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif
#include "attention.h"
#include "common/logging.h"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace sl {

MultiHeadAttention::MultiHeadAttention(const ModelConfig& config, uint32_t layer_idx)
    : n_head_(config.n_head)
    , n_kv_head_(config.n_kv_head())
    , head_dim_(config.head_dim())
    , n_embd_(config.n_embd)
    , layer_idx_(layer_idx)
{
    // Allocate weight tensors
    q_weight_ = Tensor({n_embd_, n_head_ * head_dim_}, config.weight_dtype);
    k_weight_ = Tensor({n_embd_, n_kv_head_ * head_dim_}, config.weight_dtype);
    v_weight_ = Tensor({n_embd_, n_kv_head_ * head_dim_}, config.weight_dtype);
    o_weight_ = Tensor({n_head_ * head_dim_, n_embd_}, config.weight_dtype);
    
    // Scratch buffers
    q_buf_.resize(n_head_ * head_dim_);
    k_buf_.resize(n_kv_head_ * head_dim_);
    v_buf_.resize(n_kv_head_ * head_dim_);
    attn_buf_.resize(n_head_);
    o_buf_.resize(n_embd_);
    scale_buf_.resize(n_kv_head_ * head_dim_);
}

void MultiHeadAttention::load_weights(
    const void* q_w, const void* k_w, const void* v_w, const void* o_w,
    const void* q_b, const void* k_b, const void* v_b, const void* o_b)
{
    size_t q_sz = q_weight_.nbytes();
    size_t k_sz = k_weight_.nbytes();
    size_t v_sz = v_weight_.nbytes();
    size_t o_sz = o_weight_.nbytes();
    
    memcpy(q_weight_.data(), q_w, q_sz);
    memcpy(k_weight_.data(), k_w, k_sz);
    memcpy(v_weight_.data(), v_w, v_sz);
    memcpy(o_weight_.data(), o_w, o_sz);
}

void MultiHeadAttention::compute_qkv(const float* x) {
    // Q = x @ q_weight
    simd::matvec_f32(
        static_cast<const float*>(q_weight_.data()), 
        x, q_buf_.data(), n_head_ * head_dim_, n_embd_);
    
    // K = x @ k_weight
    simd::matvec_f32(
        static_cast<const float*>(k_weight_.data()), 
        x, k_buf_.data(), n_kv_head_ * head_dim_, n_embd_);
    
    // V = x @ v_weight
    simd::matvec_f32(
        static_cast<const float*>(v_weight_.data()), 
        x, v_buf_.data(), n_kv_head_ * head_dim_, n_embd_);
}

void MultiHeadAttention::apply_rope(uint32_t position) {
    simd::rope_f32(q_buf_.data(), k_buf_.data(), 
                   head_dim_, n_head_, n_kv_head_, position, 10000.0f);
}

void MultiHeadAttention::compute_attention_scores(uint32_t position, 
                                                   KVCacheSlot* kv_cache) {
    // Store K, V into cache at current position
    size_t kv_size = n_kv_head_ * head_dim_;
    if (kv_cache) {
        kv_cache[position].k.resize(kv_size);
        kv_cache[position].v.resize(kv_size);
        memcpy(kv_cache[position].k.data(), k_buf_.data(), kv_size * sizeof(float));
        memcpy(kv_cache[position].v.data(), v_buf_.data(), kv_size * sizeof(float));
    }
    
    // For each query head, compute attention over all keys
    // If n_kv_head < n_head (GQA), expand kv heads
    float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    
    for (uint32_t h = 0; h < n_head_; h++) {
        uint32_t kv_h = (n_kv_head_ == 1) ? 0 : (h * n_kv_head_ / n_head_);
        
        // Compute attention scores: q[h] dot k[kv_h][0..position]
        // Use scratch buffer for scores
        std::vector<float> scores(position + 1);
        const float* q_h = q_buf_.data() + h * head_dim_;
        
        for (uint32_t seq = 0; seq <= position; seq++) {
            const float* k_seq = kv_cache[seq].k.data() + kv_h * head_dim_;
            scores[seq] = simd::dot_product_f32(q_h, k_seq, head_dim_) * inv_sqrt_d;
        }
        
        // Softmax
        simd::softmax_f32(scores.data(), scores.data(), position + 1);
        
        // Weighted sum of values
        float* o_h = o_buf_.data() + h * head_dim_;
        std::fill(o_h, o_h + head_dim_, 0.0f);
        for (uint32_t seq = 0; seq <= position; seq++) {
            const float* v_seq = kv_cache[seq].v.data() + kv_h * head_dim_;
            float attn_w = scores[seq];
            for (uint32_t d = 0; d < head_dim_; d++) {
                o_h[d] += attn_w * v_seq[d];
            }
        }
    }
}

void MultiHeadAttention::compute_output(float* output) {
    // output = o_buf @ o_weight
    simd::matvec_f32(
        static_cast<const float*>(o_weight_.data()), 
        o_buf_.data(), output, n_embd_, n_head_ * head_dim_);
}

void MultiHeadAttention::forward(const float* x, KVCacheSlot* kv_cache,
                                  uint32_t position, float* output) {
    compute_qkv(x);
    apply_rope(position);
    compute_attention_scores(position, kv_cache);
    compute_output(output);
}

// ====================================================================
// FlashAttention implementation (simplified)
// ====================================================================
FlashAttention::FlashAttention(uint32_t head_dim, float scale)
    : head_dim_(head_dim)
    , scale_(scale) {}

void FlashAttention::forward(const float* q, const float* k, const float* v,
                              uint32_t seq_len, uint32_t n_head, float* output) {
    float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    
    // Block-based tiled attention (simplified single-block version)
    for (uint32_t h = 0; h < n_head; h++) {
        const float* q_h = q + h * head_dim_;
        const float* k_h = k + h * head_dim_;
        const float* v_h = v + h * head_dim_;
        float* o_h = output + h * head_dim_;
        
        // Online softmax accumulator
        float max_score = -1e9f;
        float sum_exp = 0.0f;
        std::vector<float> exp_scores(seq_len);
        
        // Compute max
        for (uint32_t s = 0; s < seq_len; s++) {
            float score = simd::dot_product_f32(q_h, k_h + s * head_dim_, head_dim_) * inv_sqrt_d;
            if (score > max_score) max_score = score;
            exp_scores[s] = score;
        }
        
        // Compute exponentials and accumulate
        std::fill(o_h, o_h + head_dim_, 0.0f);
        for (uint32_t s = 0; s < seq_len; s++) {
            float exp_s = std::exp(exp_scores[s] - max_score);
            sum_exp += exp_s;
            for (uint32_t d = 0; d < head_dim_; d++) {
                o_h[d] += exp_s * v_h[s * head_dim_ + d];
            }
        }
        
        // Normalize
        for (uint32_t d = 0; d < head_dim_; d++) {
            o_h[d] /= sum_exp;
        }
    }
}

} // namespace sl