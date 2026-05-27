#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif
#include "core/tensor.h"
#include <iostream>
#include <cassert>
#include <cmath>

namespace sl {
namespace test {

void test_tensor_creation() {
    TensorShape shape{{4, 256}};
    Tensor t(shape, DType::F32);
    assert(t.numel() == 1024);
    assert(t.nbytes() == 1024 * 4);
    std::cout << "  [PASS] Tensor creation" << std::endl;
}

void test_tensor_fill() {
    TensorShape shape{{1000}};
    Tensor t(shape, DType::F32);
    t.fill(3.14f);
    
    for (size_t i = 0; i < t.numel(); i++) {
        assert(std::abs(t.at(i) - 3.14f) < 0.001f);
    }
    std::cout << "  [PASS] Tensor fill" << std::endl;
}

void test_tensor_add() {
    TensorShape shape{{256}};
    Tensor a(shape, DType::F32);
    Tensor b(shape, DType::F32);
    a.fill(1.0f);
    b.fill(2.0f);
    a.add(b);
    
    for (size_t i = 0; i < a.numel(); i++) {
        assert(std::abs(a.at(i) - 3.0f) < 0.001f);
    }
    std::cout << "  [PASS] Tensor add" << std::endl;
}

void test_simd_dot_product() {
    const size_t n = 1024;
    std::vector<float> a(n, 1.0f);
    std::vector<float> b(n, 2.0f);
    
    float result = simd::dot_product_f32(a.data(), b.data(), n);
    assert(std::abs(result - n * 2.0f) < 0.01f);
    std::cout << "  [PASS] SIMD dot product" << std::endl;
}

void test_simd_rms_norm() {
    const size_t n = 256;
    std::vector<float> input(n, 2.0f);
    std::vector<float> output(n);
    
    simd::rms_norm_f32(input.data(), output.data(), n, 1e-6f);
    
    // RMS of [2,2,...,2] = sqrt(mean(4)) = 2
    // After normalization: each element = input / 2 = 1.0
    for (size_t i = 0; i < n; i++) {
        assert(std::abs(output[i] - 1.0f) < 0.01f);
    }
    std::cout << "  [PASS] SIMD RMS norm" << std::endl;
}

void run_all_tensor_tests() {
    std::cout << "Tensor Tests:" << std::endl;
    test_tensor_creation();
    test_tensor_fill();
    test_tensor_add();
    test_simd_dot_product();
    test_simd_rms_norm();
    std::cout << "All tensor tests passed!\n" << std::endl;
}

} // namespace test
} // namespace sl