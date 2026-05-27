#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif
#include "mlp.h"
#include "common/logging.h"
#include <cstring>

namespace sl {

MLPBlock::MLPBlock(const ModelConfig& config, uint32_t layer_idx, MLPType type)
    : n_embd_(config.n_embd)
    , n_ff_(config.n_ff)
    , mlp_type_(type)
{
    gate_weight_ = Tensor({n_embd_, n_ff_}, config.weight_dtype);
    up_weight_   = Tensor({n_embd_, n_ff_}, config.weight_dtype);
    down_weight_ = Tensor({n_ff_, n_embd_}, config.weight_dtype);
    
    gate_buf_.resize(n_ff_);
    up_buf_.resize(n_ff_);
    hidden_buf_.resize(n_ff_);
}

void MLPBlock::load_weights(
    const void* gate_w, const void* up_w, const void* down_w,
    const void* /*fc1_w*/, const void* /*fc2_w*/)
{
    if (gate_w) memcpy(gate_weight_.data(), gate_w, gate_weight_.nbytes());
    if (up_w)   memcpy(up_weight_.data(), up_w, up_weight_.nbytes());
    if (down_w) memcpy(down_weight_.data(), down_w, down_weight_.nbytes());
}

size_t MLPBlock::total_weight_bytes() const {
    return gate_weight_.nbytes() + up_weight_.nbytes() + down_weight_.nbytes();
}

void MLPBlock::forward(const float* x, float* output) {
    // Gate projection: gate = x @ gate_weight
    simd::matvec_f32(
        static_cast<const float*>(gate_weight_.data()), 
        x, gate_buf_.data(), n_ff_, n_embd_);
    
    // Up projection: up = x @ up_weight
    simd::matvec_f32(
        static_cast<const float*>(up_weight_.data()), 
        x, up_buf_.data(), n_ff_, n_embd_);
    
    // SiLU activation on gate
    simd::silu_f32(gate_buf_.data(), gate_buf_.data(), n_ff_);
    
    // Element-wise multiply: hidden = gate * up
    simd::gate_mul_f32(gate_buf_.data(), up_buf_.data(), hidden_buf_.data(), n_ff_);
    
    // Down projection: output = hidden @ down_weight
    simd::matvec_f32(
        static_cast<const float*>(down_weight_.data()), 
        hidden_buf_.data(), output, n_embd_, n_ff_);
}

} // namespace sl