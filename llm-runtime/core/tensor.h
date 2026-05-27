#pragma once
#include "common/platform.h"

#include "common/types.h"
#include <memory>
#include <immintrin.h>
#include <cassert>
#include <initializer_list>

namespace sl {

class PagingManager;

class Tensor {
public:
    Tensor();
    Tensor(const TensorShape& shape, DType dtype);
    Tensor(std::initializer_list<size_t> dims, DType dtype);
    ~Tensor();

    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor clone() const;

    const TensorShape& shape() const { return shape_; }
    DType dtype() const { return dtype_; }
    size_t nbytes() const;
    size_t numel() const { return shape_.numel(); }

    void* data();
    const void* data() const;
    template<typename T> T* data_ptr() { return static_cast<T*>(data()); }
    template<typename T> const T* data_ptr() const { return static_cast<const T*>(data()); }

    float at(size_t idx) const;
    void set_at(size_t idx, float val);

    bool is_resident() const { return resident_ != nullptr; }
    void ensure_resident();
    void evict();

    void set_page_id(uint64_t id) { page_id_ = id; }
    uint64_t page_id() const { return page_id_; }

    void fill(float value);
    void zero();
    void scale(float factor);
    void add(const Tensor& other);
    void add_scaled(const Tensor& other, float scale);

    static Tensor zeros(const TensorShape& shape, DType dtype);
    static Tensor ones(const TensorShape& shape, DType dtype);

    void quantize(DType target_dtype);
    void dequantize();

private:
    TensorShape shape_;
    DType dtype_ = DType::F32;
    std::unique_ptr<uint8_t[]> owner_;
    void* resident_ = nullptr;
    uint64_t page_id_ = 0;
    bool quantized_ = false;

    void allocate();
    void deallocate();
};

namespace simd {
float dot_product_f32(const float* a, const float* b, size_t n);
void rms_norm_f32(const float* input, float* output, size_t n, float eps);
void softmax_f32(const float* input, float* output, size_t n);
void matvec_f32(const float* A, const float* x, float* y, size_t M, size_t N);
void rope_f32(float* q, float* k, size_t head_dim, size_t n_head,
              size_t n_kv_head, size_t seq_pos, float theta);
void silu_f32(const float* input, float* output, size_t n);
void gate_mul_f32(const float* gate, const float* up, float* output, size_t n);
} // namespace simd

} // namespace sl