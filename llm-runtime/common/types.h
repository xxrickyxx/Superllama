#pragma once
#include "common/platform.h"

#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <optional>
#include <variant>
#include <functional>
#include <memory>
#include <chrono>
#include <span>

namespace sl {

// ====================================================================
// Precision / Quantization types
// ====================================================================
enum class DType : uint8_t {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    Q4_0 = 3,   // 4-bit GGUF style, group size 32
    Q4_1 = 4,
    Q5_0 = 5,
    Q5_1 = 6,
    Q8_0 = 7,
    Q8_1 = 8,
    Q2_K = 9,
    Q3_K = 10,
    Q4_K = 11,
    Q5_K = 12,
    Q6_K = 13,
    Q8_K = 14,
    IQ4_NL = 15,
    COUNT
};

inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::F32: return 4;
        case DType::F16: case DType::BF16: return 2;
        case DType::Q4_0: case DType::Q4_1: return 0; // needs block info
        case DType::Q8_0: case DType::Q8_1: return 0;
        default: return 0;
    }
}

inline const char* dtype_name(DType dt) {
    switch (dt) {
        case DType::F32: return "f32";
        case DType::F16: return "f16";
        case DType::BF16: return "bf16";
        case DType::Q4_0: return "q4_0";
        case DType::Q4_1: return "q4_1";
        case DType::Q5_0: return "q5_0";
        case DType::Q5_1: return "q5_1";
        case DType::Q8_0: return "q8_0";
        case DType::Q4_K: return "q4_K";
        case DType::Q5_K: return "q5_K";
        case DType::Q6_K: return "q6_K";
        case DType::Q8_K: return "q8_K";
        default: return "unknown";
    }
}

// ====================================================================
// Tensor shape & descriptor
// ====================================================================
struct TensorShape {
    std::vector<size_t> dims;
    
    size_t rank() const { return dims.size(); }
    size_t numel() const {
        size_t n = 1;
        for (auto d : dims) n *= d;
        return n;
    }
    size_t operator[](size_t i) const { return dims[i]; }
};

// ====================================================================
// Model architecture enumeration
// ====================================================================
enum class ModelArch : uint8_t {
    UNKNOWN = 0,
    LLAMA,
    MISTRAL,
    FALCON,
    GEMMA,
    PHI,
    QWEN,
    DEEPSEEK,
    CUSTOM
};

inline const char* arch_name(ModelArch a) {
    switch (a) {
        case ModelArch::LLAMA: return "llama";
        case ModelArch::MISTRAL: return "mistral";
        case ModelArch::FALCON: return "falcon";
        case ModelArch::GEMMA: return "gemma";
        case ModelArch::PHI: return "phi";
        case ModelArch::QWEN: return "qwen";
        case ModelArch::DEEPSEEK: return "deepseek";
        default: return "unknown";
    }
}

// ====================================================================
// Model configuration (hyperparameters)
// ====================================================================
struct ModelConfig {
    std::string name;
    std::string path;           // file path on disk
    ModelArch architecture = ModelArch::UNKNOWN;
    
    // Hyperparameters
    uint32_t n_vocab = 32000;
    uint32_t n_embd = 4096;
    uint32_t n_head = 32;
    uint32_t n_head_kv = 32;
    uint32_t n_layer = 32;
    uint32_t n_rot = 128;       // rope dimension
    uint32_t n_ff = 14336;      // feed-forward inner dim
    float f_norm_eps = 1e-5f;
    uint32_t n_ctx = 4096;      // context length
    
    // Quantization
    DType weight_dtype = DType::F16;
    uint32_t block_size = 256;
    
    // Model size info
    uint64_t total_params = 0;
    uint64_t file_size_bytes = 0;
    
    // Tokenizer
    std::string tokenizer_path;
    bool add_bos = true;
    bool add_eos = false;
    uint32_t bos_token_id = 1;
    uint32_t eos_token_id = 2;
    
    // Derived
    uint32_t head_dim() const { return n_embd / n_head; }
    uint32_t n_kv_head() const { return n_head_kv > 0 ? n_head_kv : n_head; }
};

// ====================================================================
// Token and sequence types
// ====================================================================
using Token = int32_t;
using TokenSequence = std::vector<Token>;

// ====================================================================
// KV Cache slot
// ====================================================================
struct KVCacheSlot {
    std::vector<float> k;   // [n_head_kv, head_dim]
    std::vector<float> v;   // [n_head_kv, head_dim]
};

// ====================================================================
// Generation parameters
// ====================================================================
struct GenerationConfig {
    uint32_t max_tokens = 256;
    float temperature = 0.7f;
    float top_p = 0.9f;
    float top_k = 40;
    float repetition_penalty = 1.1f;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    uint32_t seed = 0xFFFFFFFF;
    bool stream = true;
    std::vector<std::string> stop_strings;
    std::vector<Token> stop_tokens;
};

// ====================================================================
// Inference request/response
// ====================================================================
struct InferenceRequest {
    std::string model_name;
    std::string prompt;
    std::optional<std::string> system_prompt;
    GenerationConfig gen_config;
    
    // For chat mode
    struct Message {
        std::string role;       // "user", "assistant", "system"
        std::string content;
    };
    std::vector<Message> messages;
    bool chat_mode = false;
};

struct InferenceToken {
    Token token_id;
    std::string text;
    float log_prob = 0.0f;
    bool is_final = false;
};

using TokenCallback = std::function<void(const InferenceToken&)>;

// ====================================================================
// Model entry in registry
// ====================================================================
struct ModelEntry {
    std::string name;
    std::string path;
    std::string family;
    ModelArch architecture;
    DType quantization;
    uint64_t size_bytes;
    uint64_t param_count;
    std::string digest;         // SHA256
    std::chrono::system_clock::time_point modified_at;
    
    std::string size_str() const {
        char buf[64];
        double gb = size_bytes / (1024.0 * 1024.0 * 1024.0);
        snprintf(buf, sizeof(buf), "%.1f GB", gb);
        return buf;
    }
};

// ====================================================================
// Download progress
// ====================================================================
struct DownloadProgress {
    std::string model_name;
    uint64_t total_bytes = 0;
    uint64_t downloaded_bytes = 0;
    double speed_mbps = 0.0;
    int eta_seconds = 0;
    bool completed = false;
    std::string status;         // "downloading", "extracting", "complete", "error"
    std::string error_message;
};

using DownloadProgressCallback = std::function<void(const DownloadProgress&)>;

// ====================================================================
// Memory paging statistics
// ====================================================================
struct PagingStats {
    uint64_t total_model_bytes = 0;
    uint64_t resident_bytes = 0;
    uint64_t mapped_bytes = 0;
    uint64_t cache_hits = 0;
    uint64_t cache_misses = 0;
    uint64_t prefetch_hits = 0;
    uint64_t io_bytes_read = 0;
    double cache_hit_rate = 0.0;
    uint32_t active_layers = 0;
    uint32_t total_layers = 0;
};

// ====================================================================
// Server configuration
// ====================================================================
struct ServerConfig {
    std::string host = "127.0.0.1";
    uint16_t port = 11434;
    uint32_t num_threads = 0;   // 0 = auto
    bool enable_cors = true;
    uint32_t max_queue_size = 32;
    std::vector<std::string> model_paths = {
        "./models",
        "%USERPROFILE%/.cache/superllama/models"
    };
};

// ====================================================================
// Logging levels
// ====================================================================
enum class LogLevel : uint8_t {
    TRACE = 0,
    DEBUG = 1,
    INFO = 2,
    WARN = 3,
    ERROR = 4,
    FATAL = 5
};

}  // namespace sl