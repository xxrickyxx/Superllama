#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif
#include "tensor.h"
#include "common/logging.h"
#include <algorithm>
#include <cmath>
#include <immintrin.h>

namespace sl {

// ====================================================================
// Tensor implementation
// ====================================================================
// ============ FP16 helpers (must come before usage in Tensor methods) ============
inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) {
        if (mant == 0) { uint32_t r = sign; return *reinterpret_cast<float*>(&r); }
        uint32_t norm_exp = 127 - 14;
        while ((mant & 0x400) == 0) { mant <<= 1; norm_exp--; }
        mant &= 0x3FF;
        uint32_t f32 = sign | (norm_exp << 23) | (mant << 13);
        return *reinterpret_cast<float*>(&f32);
    } else if (exp == 0x1F) {
        uint32_t f32 = sign | (0xFF << 23) | (mant << 13);
        return *reinterpret_cast<float*>(&f32);
    } else {
        uint32_t f32 = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        return *reinterpret_cast<float*>(&f32);
    }
}
inline uint16_t fp32_to_fp16(float f) {
    uint32_t x = *reinterpret_cast<uint32_t*>(&f);
    uint32_t sign = x >> 31;
    int32_t exp = ((x >> 23) & 0xFF) - 127;
    uint32_t mant = x & 0x7FFFFF;
    if (exp > 15) return (sign << 15) | 0x7C00;
    if (exp < -14) return (sign << 15);
    if (exp <= -1) {
        mant = (mant | 0x800000) >> (-exp);
        return (sign << 15) | ((mant + 0x1000) >> 13);
    }
    return (sign << 15) | ((exp + 15) << 10) | (mant >> 13);
}
// ================================================================

Tensor::Tensor() : shape_({}), dtype_(DType::F32), resident_(nullptr) {}

Tensor::Tensor(const TensorShape& shape, DType dtype)
    : shape_(shape), dtype_(dtype) {
    allocate();
}

Tensor::Tensor(std::initializer_list<size_t> dims, DType dtype)
    : shape_(), dtype_(dtype) {
    shape_.dims.assign(dims.begin(), dims.end());
    allocate();
}

Tensor::~Tensor() {
    deallocate();
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_))
    , dtype_(other.dtype_)
    , owner_(std::move(other.owner_))
    , resident_(other.resident_)
    , page_id_(other.page_id_)
    , quantized_(other.quantized_) {
    other.resident_ = nullptr;
    other.page_id_ = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        deallocate();
        shape_ = std::move(other.shape_);
        dtype_ = other.dtype_;
        owner_ = std::move(other.owner_);
        resident_ = other.resident_;
        page_id_ = other.page_id_;
        quantized_ = other.quantized_;
        other.resident_ = nullptr;
        other.page_id_ = 0;
    }
    return *this;
}

Tensor Tensor::clone() const {
    Tensor result(shape_, dtype_);
    if (resident_ && result.resident_) {
        memcpy(result.resident_, resident_, nbytes());
    }
    result.quantized_ = quantized_;
    return result;
}

size_t Tensor::nbytes() const {
    size_t n = shape_.numel();
    switch (dtype_) {
        case DType::F32: return n * 4;
        case DType::F16: case DType::BF16: return n * 2;
        case DType::Q4_0: return n / 2 + n / 32; // approximate
        case DType::Q8_0: return n + n / 32;
        default: return n * 4;
    }
}

void Tensor::allocate() {
    size_t bytes = nbytes();
    if (bytes > 0) {
        owner_ = std::make_unique<uint8_t[]>(bytes);
        resident_ = owner_.get();
        memset(resident_, 0, bytes);
    }
}

void Tensor::deallocate() {
    owner_.reset();
    resident_ = nullptr;
}

void* Tensor::data() {
    ensure_resident();
    return resident_;
}

const void* Tensor::data() const {
    return resident_;
}

void Tensor::ensure_resident() {
    if (!resident_) {
        allocate();
    }
}

void Tensor::evict() {
    // In a real system, this would coordinate with PagingManager
    // to ensure data is backed to SSD before eviction
    if (page_id_ != 0) {
        // Data is backed by SSD mapping, safe to evict
        owner_.reset();
        resident_ = nullptr;
    }
}

float Tensor::at(size_t idx) const {
    if (!resident_) return 0.0f;
    
    switch (dtype_) {
        case DType::F32: return static_cast<const float*>(resident_)[idx];
        case DType::F16: {
            // Simplified FP16 conversion
            uint16_t val = static_cast<const uint16_t*>(resident_)[idx];
            return fp16_to_fp32(val);
        }
        default: return 0.0f;
    }
}

void Tensor::set_at(size_t idx, float val) {
    ensure_resident();
    switch (dtype_) {
        case DType::F32: static_cast<float*>(resident_)[idx] = val; break;
        case DType::F16: static_cast<uint16_t*>(resident_)[idx] = fp32_to_fp16(val); break;
        default: break;
    }
}

void Tensor::fill(float value) {
    ensure_resident();
    size_t n = numel();
    switch (dtype_) {
        case DType::F32: {
            auto* ptr = static_cast<float*>(resident_);
            std::fill(ptr, ptr + n, value);
            break;
        }
        default: {
            for (size_t i = 0; i < n; i++) set_at(i, value);
            break;
        }
    }
}

void Tensor::zero() { fill(0.0f); }

void Tensor::scale(float factor) {
    ensure_resident();
    size_t n = numel();
    if (dtype_ == DType::F32) {
        auto* ptr = static_cast<float*>(resident_);
        __m256 f = _mm256_set1_ps(factor);
        for (size_t i = 0; i + 8 <= n; i += 8) {
            __m256 v = _mm256_loadu_ps(ptr + i);
            _mm256_storeu_ps(ptr + i, _mm256_mul_ps(v, f));
        }
        for (size_t i = n - (n % 8); i < n; i++) ptr[i] *= factor;
    }
}

void Tensor::add(const Tensor& other) {
    ensure_resident();
    assert(shape_.numel() == other.shape_.numel());
    size_t n = numel();
    if (dtype_ == DType::F32 && other.dtype_ == DType::F32) {
        auto* a = static_cast<float*>(resident_);
        const auto* b = static_cast<const float*>(other.resident_);
        for (size_t i = 0; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            _mm256_storeu_ps(a + i, _mm256_add_ps(va, vb));
        }
        for (size_t i = n - (n % 8); i < n; i++) a[i] += b[i];
    }
}

void Tensor::add_scaled(const Tensor& other, float scale) {
    ensure_resident();
    assert(shape_.numel() == other.shape_.numel());
    size_t n = numel();
    if (dtype_ == DType::F32 && other.dtype_ == DType::F32) {
        auto* a = static_cast<float*>(resident_);
        const auto* b = static_cast<const float*>(other.resident_);
        __m256 sc = _mm256_set1_ps(scale);
        for (size_t i = 0; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_mul_ps(_mm256_loadu_ps(b + i), sc);
            _mm256_storeu_ps(a + i, _mm256_add_ps(va, vb));
        }
        for (size_t i = n - (n % 8); i < n; i++) a[i] += b[i] * scale;
    }
}

// (FP16 helpers were moved above Tensor methods to prevent forward-reference errors)

// ====================================================================
// SIMD operations
// ====================================================================
namespace simd {

float dot_product_f32(const float* a, const float* b, size_t n) {
    __m256 sum8 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum8 = _mm256_fmadd_ps(va, vb, sum8);
    }
    // Horizontal sum
    __m128 lo = _mm256_castps256_ps128(sum8);
    __m128 hi = _mm256_extractf128_ps(sum8, 1);
    __m128 sum4 = _mm_add_ps(lo, hi);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    float result = _mm_cvtss_f32(sum4);
    
    // Remainder
    for (; i < n; i++) result += a[i] * b[i];
    return result;
}

void rms_norm_f32(const float* input, float* output, size_t n, float eps) {
    // Compute mean square
    __m256 ms8 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(input + i);
        ms8 = _mm256_fmadd_ps(v, v, ms8);
    }
    float ms = 0.0f;
    {
        __m128 lo = _mm256_castps256_ps128(ms8);
        __m128 hi = _mm256_extractf128_ps(ms8, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        ms = _mm_cvtss_f32(lo);
    }
    for (; i < n; i++) ms += input[i] * input[i];
    
    float rms = 1.0f / std::sqrt(ms / n + eps);
    __m256 scale = _mm256_set1_ps(rms);
    
    i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(input + i);
        _mm256_storeu_ps(output + i, _mm256_mul_ps(v, scale));
    }
    for (; i < n; i++) output[i] = input[i] * rms;
}

void softmax_f32(const float* input, float* output, size_t n) {
    // Find max for numerical stability
    float max_val = input[0];
    for (size_t i = 1; i < n; i++) max_val = std::max(max_val, input[i]);
    
    // Compute exponentials and sum
    float sum = 0.0f;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(input + i);
        v = _mm256_sub_ps(v, _mm256_set1_ps(max_val));
        // Approximate exp using Intel SVML or custom
        float tmp[8];
        _mm256_storeu_ps(tmp, v);
        for (int j = 0; j < 8; j++) {
            tmp[j] = std::exp(tmp[j]);
            sum += tmp[j];
        }
        _mm256_storeu_ps(output + i, _mm256_loadu_ps(tmp));
    }
    for (; i < n; i++) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    
    // Normalize
    float inv_sum = 1.0f / sum;
    i = 0;
    __m256 is = _mm256_set1_ps(inv_sum);
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(output + i);
        _mm256_storeu_ps(output + i, _mm256_mul_ps(v, is));
    }
    for (; i < n; i++) output[i] *= inv_sum;
}

// ... simplified implementations for remaining functions (full versions would be ~2000 lines)

void silu_f32(const float* input, float* output, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float x = input[i];
        output[i] = x * (1.0f / (1.0f + std::exp(-x)));
    }
}

void gate_mul_f32(const float* gate, const float* up, float* output, size_t n) {
    // output = gate * up (element-wise) where gate was silu-activated
    for (size_t i = 0; i + 8 <= n; i += 8) {
        __m256 g = _mm256_loadu_ps(gate + i);
        __m256 u = _mm256_loadu_ps(up + i);
        _mm256_storeu_ps(output + i, _mm256_mul_ps(g, u));
    }
    for (size_t i = n - (n % 8); i < n; i++) {
        output[i] = gate[i] * up[i];
    }
}

void rope_f32(float* q, float* k, size_t head_dim, size_t n_head, 
              size_t n_kv_head, size_t seq_pos, float theta) {
    // Apply rotary position embeddings
    size_t n_half = head_dim / 2;
    for (size_t h = 0; h < n_head; h++) {
        for (size_t j = 0; j < n_half; j++) {
            float freq = 1.0f / std::pow(theta, 2.0f * j / head_dim);
            float val = static_cast<float>(seq_pos) * freq;
            float cos_val = std::cos(val);
            float sin_val = std::sin(val);
            
            size_t offset = h * head_dim;
            float q0 = q[offset + j];
            float q1 = q[offset + j + n_half];
            q[offset + j] = q0 * cos_val - q1 * sin_val;
            q[offset + j + n_half] = q1 * cos_val + q0 * sin_val;
        }
    }
    // KV heads similar
    for (size_t h = 0; h < n_kv_head; h++) {
        for (size_t j = 0; j < n_half; j++) {
            float freq = 1.0f / std::pow(theta, 2.0f * j / head_dim);
            float val = static_cast<float>(seq_pos) * freq;
            float cos_val = std::cos(val);
            float sin_val = std::sin(val);
            
            size_t offset = h * head_dim;
            float k0 = k[offset + j];
            float k1 = k[offset + j + n_half];
            k[offset + j] = k0 * cos_val - k1 * sin_val;
            k[offset + j + n_half] = k1 * cos_val + k0 * sin_val;
        }
    }
}

void matvec_f32(const float* A, const float* x, float* y, size_t M, size_t N) {
    for (size_t i = 0; i < M; i++) {
        y[i] = dot_product_f32(A + i * N, x, N);
    }
}

} // namespace simd

} // namespace sl