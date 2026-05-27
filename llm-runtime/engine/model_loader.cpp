#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif
#include "model_loader.h"
#include "common/logging.h"
#include <cstring>
#include <algorithm>
#include <regex>
#include <sstream>

namespace sl {

// ====================================================================
// GGUF Magic (little-endian)
// ====================================================================
static constexpr uint32_t GGUF_MAGIC = 0x46554747;  // "GGUF"

// ====================================================================
// GGUFReader Implementation
// ====================================================================
GGUFReader::GGUFReader(const std::string& filepath)
    : filepath_(filepath) {
    
    file_handle_ = CreateFileA(
        filepath.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS,
        nullptr
    );
    
    if (file_handle_ == INVALID_HANDLE_VALUE) {
        SL_LOG_ERROR("Cannot open model file: {}", filepath);
        return;
    }
    
    LARGE_INTEGER li;
    GetFileSizeEx(file_handle_, &li);
    file_size_ = li.QuadPart;
    
    mapping_handle_ = CreateFileMappingA(file_handle_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapping_handle_) {
        mapped_base_ = MapViewOfFile(mapping_handle_, FILE_MAP_READ, 0, 0, 0);
    }
    
    if (!parse_header()) {
        SL_LOG_ERROR("Failed to parse GGUF header");
        return;
    }
    
    valid_ = true;
    SL_LOG_INFO("GGUF model opened: {} version={}, {} tensors, {} metadata pairs",
                filepath, gguf_version_, tensor_count_, metadata_count_);
}

GGUFReader::~GGUFReader() {
    if (mapped_base_) UnmapViewOfFile(mapped_base_);
    if (mapping_handle_) CloseHandle(mapping_handle_);
    if (file_handle_ != INVALID_HANDLE_VALUE) CloseHandle(file_handle_);
}

bool GGUFReader::parse_header() {
    if (!mapped_base_) return false;
    
    auto* base = static_cast<uint8_t*>(mapped_base_);
    uint64_t offset = 0;
    
    // Magic
    uint32_t magic;
    memcpy(&magic, base, 4);
    if (magic != GGUF_MAGIC) {
        SL_LOG_WARN("Not a GGUF file (magic mismatch), attempting custom format...");
        return false;  // Could implement fallback custom format
    }
    offset += 4;
    
    // Version
    memcpy(&gguf_version_, base + offset, 4);
    offset += 4;
    
    // Tensor count
    memcpy(&tensor_count_, base + offset, 8);
    offset += 8;
    
    // Metadata count
    memcpy(&metadata_count_, base + offset, 8);
    offset += 8;
    
    // Parse metadata K-V pairs
    for (uint64_t i = 0; i < metadata_count_; i++) {
        std::string key = read_string_at(offset);
        
        // Read value type
        uint32_t value_type;
        memcpy(&value_type, base + offset, 4);
        offset += 4;
        
        // Read value based on type
        std::string value_str;
        switch (value_type) {
            case GGUFType::U32: {
                uint32_t v;
                memcpy(&v, base + offset, 4);
                offset += 4;
                value_str = std::to_string(v);
                break;
            }
            case GGUFType::F32: {
                float v;
                memcpy(&v, base + offset, 4);
                offset += 4;
                value_str = std::to_string(v);
                break;
            }
            case GGUFType::STRING: {
                value_str = read_string_at(offset);
                break;
            }
            case GGUFType::U64: {
                uint64_t v;
                memcpy(&v, base + offset, 8);
                offset += 8;
                value_str = std::to_string(v);
                break;
            }
            case GGUFType::I32: {
                int32_t v;
                memcpy(&v, base + offset, 4);
                offset += 4;
                value_str = std::to_string(v);
                break;
            }
            case GGUFType::BOOL: {
                uint8_t v;
                memcpy(&v, base + offset, 1);
                offset += 1;
                value_str = v ? "true" : "false";
                break;
            }
            case GGUFType::ARRAY: {
                // Skip array type for now
                uint32_t array_type;
                memcpy(&array_type, base + offset, 4);
                offset += 4;
                uint64_t array_len;
                memcpy(&array_len, base + offset, 8);
                offset += 8;
                // Skip array elements
                for (uint64_t j = 0; j < array_len; j++) {
                    if (array_type == GGUFType::U32) offset += 4;
                    else if (array_type == GGUFType::F32) offset += 4;
                    else if (array_type == GGUFType::STRING) { read_string_at(offset); }
                    else offset += 4;
                }
                value_str = "[...]";
                break;
            }
            default:
                SL_LOG_WARN("Unknown GGUF value type: {}", value_type);
                offset += 4;
                break;
        }
        
        metadata_[key] = value_str;
    }
    
    SL_LOG_DEBUG("GGUF metadata parsed: {} entries", metadata_count_);
    return true;
}

std::string GGUFReader::read_string_at(uint64_t& offset) {
    auto* base = static_cast<uint8_t*>(mapped_base_);
    
    uint64_t len;
    memcpy(&len, base + offset, 8);
    offset += 8;
    
    std::string result(reinterpret_cast<char*>(base + offset), len);
    offset += len;
    return result;
}

ModelConfig GGUFReader::read_config() {
    ModelConfig cfg;
    
    // Extract from metadata
    auto get_meta_u32 = [&](const std::string& key) -> uint32_t {
        auto it = metadata_.find(key);
        return it != metadata_.end() ? std::stoul(it->second) : 0;
    };
    
    auto get_meta_float = [&](const std::string& key) -> float {
        auto it = metadata_.find(key);
        return it != metadata_.end() ? std::stof(it->second) : 0.0f;
    };
    
    auto get_meta_str = [&](const std::string& key) -> std::string {
        auto it = metadata_.find(key);
        return it != metadata_.end() ? it->second : "";
    };
    
    cfg.name = get_meta_str("general.name");
    if (cfg.name.empty()) cfg.name = std::filesystem::path(filepath_).stem().string();
    
    // Architecture detection
    std::string arch_str = get_meta_str("general.architecture");
    if (arch_str.find("llama") != std::string::npos) cfg.architecture = ModelArch::LLAMA;
    else if (arch_str.find("mistral") != std::string::npos) cfg.architecture = ModelArch::MISTRAL;
    else if (arch_str.find("gemma") != std::string::npos) cfg.architecture = ModelArch::GEMMA;
    else if (arch_str.find("phi") != std::string::npos) cfg.architecture = ModelArch::PHI;
    else if (arch_str.find("qwen") != std::string::npos) cfg.architecture = ModelArch::QWEN;
    else cfg.architecture = ModelArch::UNKNOWN;
    
    // Hyperparameters
    cfg.n_vocab = get_meta_u32("llama.vocab_size");
    cfg.n_embd = get_meta_u32("llama.embedding_length");
    cfg.n_head = get_meta_u32("llama.attention.head_count");
    cfg.n_head_kv = get_meta_u32("llama.attention.head_count_kv");
    cfg.n_layer = get_meta_u32("llama.block_count");
    cfg.n_ff = get_meta_u32("llama.feed_forward_length");
    cfg.n_ctx = get_meta_u32("llama.context_length");
    cfg.f_norm_eps = get_meta_float("llama.attention.layer_norm_rms_epsilon");
    
    // If n_head_kv wasn't specified, default to n_head
    if (cfg.n_head_kv == 0) cfg.n_head_kv = cfg.n_head;
    
    // Tokenizer info
    cfg.bos_token_id = get_meta_u32("tokenizer.ggml.bos_token_id");
    cfg.eos_token_id = get_meta_u32("tokenizer.ggml.eos_token_id");
    if (cfg.bos_token_id == 0) cfg.bos_token_id = 1;
    if (cfg.eos_token_id == 0) cfg.eos_token_id = 2;
    
    cfg.path = filepath_;
    cfg.file_size_bytes = file_size_;
    
    // Detect quantization from file name and metadata
    std::string file_str = get_meta_str("general.file_type");
    if (file_str.find("Q4_0") != std::string::npos) cfg.weight_dtype = DType::Q4_0;
    else if (file_str.find("Q4_K") != std::string::npos) cfg.weight_dtype = DType::Q4_K;
    else if (file_str.find("Q5_K") != std::string::npos) cfg.weight_dtype = DType::Q5_K;
    else if (file_str.find("Q8_0") != std::string::npos) cfg.weight_dtype = DType::Q8_0;
    else if (file_str.find("F16") != std::string::npos) cfg.weight_dtype = DType::F16;
    else cfg.weight_dtype = DType::F32;
    
    return cfg;
}

std::vector<GGUFReader::TensorInfo> GGUFReader::enumerate_tensors() {
    std::vector<TensorInfo> tensors;
    
    if (!valid_) return tensors;
    
    auto* base = static_cast<uint8_t*>(mapped_base_);
    
    // Calculate offset where tensor infos begin (after metadata)
    // We need to re-parse or store the offset during header parsing
    // Simplified: scan through the file after metadata
    uint64_t offset = 4 + 4 + 8 + 8; // magic + version + tensor_count + metadata_count
    
    // Skip metadata
    for (uint64_t i = 0; i < metadata_count_; i++) {
        read_string_at(offset); // key
        uint32_t value_type;
        memcpy(&value_type, base + offset, 4);
        offset += 4;
        // Skip value (simplified - production code would handle all types)
        switch (value_type) {
            case GGUFType::STRING: read_string_at(offset); break;
            case GGUFType::U32: offset += 4; break;
            case GGUFType::U64: offset += 8; break;
            case GGUFType::F32: offset += 4; break;
            case GGUFType::I32: offset += 4; break;
            case GGUFType::BOOL: offset += 1; break;
            default: offset += 4; break;
        }
    }
    
    // Now at tensor info section
    for (uint64_t i = 0; i < tensor_count_; i++) {
        TensorInfo info;
        info.name = read_string_at(offset);
        
        // Number of dimensions
        uint32_t n_dims;
        memcpy(&n_dims, base + offset, 4);
        offset += 4;
        
        // Dimension sizes
        info.shape.dims.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; d++) {
            uint64_t dim;
            memcpy(&dim, base + offset, 8);
            offset += 8;
            info.shape.dims[d] = dim;
        }
        
        // Data type
        uint32_t ggml_type;
        memcpy(&ggml_type, base + offset, 4);
        offset += 4;
        
        switch (ggml_type) {
            case 0: info.dtype = DType::F32; break;
            case 1: info.dtype = DType::F16; break;
            case 2: info.dtype = DType::Q4_0; break;
            case 3: info.dtype = DType::Q4_1; break;
            case 7: info.dtype = DType::Q8_0; break;
            case 10: info.dtype = DType::Q2_K; break;
            case 11: info.dtype = DType::Q3_K; break;
            case 12: info.dtype = DType::Q4_K; break;
            case 13: info.dtype = DType::Q5_K; break;
            case 14: info.dtype = DType::Q6_K; break;
            default: info.dtype = DType::F32; break;
        }
        
        // Data offset
        memcpy(&info.offset, base + offset, 8);
        offset += 8;
        
        // Compute size
        info.size_bytes = info.shape.numel();
        switch (info.dtype) {
            case DType::F32: info.size_bytes *= 4; break;
            case DType::F16: info.size_bytes *= 2; break;
            case DType::Q4_0: info.size_bytes = info.size_bytes / 2 + info.size_bytes / 32; break;
            case DType::Q8_0: info.size_bytes += info.size_bytes / 32; break;
            default: info.size_bytes *= 4; break;
        }
        
        tensors.push_back(info);
    }
    
    SL_LOG_INFO("Enumerated {} tensors", tensors.size());
    return tensors;
}

void GGUFReader::read_tensor_data(uint64_t offset, void* buffer, uint64_t size) {
    if (mapped_base_) {
        memcpy(buffer, static_cast<uint8_t*>(mapped_base_) + offset, size);
    }
}

void* GGUFReader::map_tensor_region(uint64_t offset, uint64_t size) {
    if (mapped_base_) {
        return static_cast<uint8_t*>(mapped_base_) + offset;
    }
    return nullptr;
}

uint64_t GGUFReader::file_size() const {
    return file_size_;
}

void GGUFReader::classify_tensor(const std::string& name, 
                                  uint32_t& layer_idx,
                                  uint32_t& tensor_type, 
                                  std::string& canonical_name) {
    // Parse GGUF naming convention: "blk.{N}.{type}.{wtype}.weight"
    // e.g., "blk.0.attn_q.weight" → layer=0, attn, q_weight
    
    static std::regex blk_regex(R"(blk\.(\d+)\.(.+))");
    std::smatch match;
    
    if (std::regex_search(name, match, blk_regex)) {
        layer_idx = std::stoul(match[1].str());
        std::string rest = match[2].str();
        
        if (rest.find("attn_q") != std::string::npos) {
            tensor_type = 0; canonical_name = "q_weight";
        } else if (rest.find("attn_k") != std::string::npos) {
            tensor_type = 0; canonical_name = "k_weight";
        } else if (rest.find("attn_v") != std::string::npos) {
            tensor_type = 0; canonical_name = "v_weight";
        } else if (rest.find("attn_output") != std::string::npos) {
            tensor_type = 0; canonical_name = "o_weight";
        } else if (rest.find("ffn_gate") != std::string::npos) {
            tensor_type = 1; canonical_name = "gate_weight";
        } else if (rest.find("ffn_up") != std::string::npos) {
            tensor_type = 1; canonical_name = "up_weight";
        } else if (rest.find("ffn_down") != std::string::npos) {
            tensor_type = 1; canonical_name = "down_weight";
        } else if (rest.find("attn_norm") != std::string::npos) {
            tensor_type = 2; canonical_name = "attn_norm";
        } else if (rest.find("ffn_norm") != std::string::npos) {
            tensor_type = 2; canonical_name = "ffn_norm";
        } else {
            tensor_type = 3; canonical_name = rest;  // unknown
        }
    } else {
        layer_idx = 0xFFFFFFFF;  // Global tensor (embedding, lm_head, etc.)
        tensor_type = 3;
        
        if (name.find("token_embd") != std::string::npos) {
            tensor_type = 3; canonical_name = "token_embd";
        } else if (name.find("output") != std::string::npos || 
                   name.find("lm_head") != std::string::npos) {
            tensor_type = 4; canonical_name = "lm_head";
        } else {
            canonical_name = name;
        }
    }
}

// ====================================================================
// ModelLoader Implementation
// ====================================================================
ModelLoader::ModelLoader(SSDPagingManager& paging_mgr)
    : paging_mgr_(paging_mgr) {}

ModelLoader::~ModelLoader() {
    unload();
}

ModelConfig ModelLoader::load(const std::string& model_path) {
    unload();
    
    GGUFReader reader(model_path);
    if (!reader.is_valid()) {
        SL_LOG_ERROR("Invalid GGUF file: {}", model_path);
        return {};
    }
    
    config_ = reader.read_config();
    config_.path = model_path;
    
    // Open in paging manager
    if (!paging_mgr_.open_model(model_path)) {
        SL_LOG_ERROR("Paging manager failed to open model");
        return {};
    }
    paging_mgr_.set_model_config(config_);
    
    // Enumerate and register all tensors
    auto tensors = reader.enumerate_tensors();
    layer_tensors_.resize(config_.n_layer);
    
    SL_LOG_INFO("Registering {} tensors across {} layers...", 
                tensors.size(), config_.n_layer);
    
    for (const auto& t : tensors) {
        uint32_t layer_idx;
        uint32_t tensor_type;
        std::string canonical_name;
        
        reader.classify_tensor(t.name, layer_idx, tensor_type, canonical_name);
        
        // Build full key for tensor lookup
        std::string full_key;
        if (layer_idx != 0xFFFFFFFF) {
            full_key = std::format("layer{}.{}", layer_idx, canonical_name);
        } else {
            full_key = canonical_name;
        }
        
        register_tensor_with_paging(full_key, layer_idx, tensor_type,
                                     t.offset, t.size_bytes, t.dtype);
    }
    
    // Prefetch embeddings and norms (small, always needed)
    prefetch_embeddings();
    prefetch_all_norms();
    
    loaded_ = true;
    SL_LOG_INFO("Model loaded: {} ({} layers, {} params, {:.1f} GB)",
                config_.name, config_.n_layer, 
                config_.total_params, config_.file_size_bytes / 1e9);
    
    return config_;
}

void ModelLoader::unload() {
    paging_mgr_.close();
    tensor_pages_.clear();
    layer_tensors_.clear();
    loaded_ = false;
}

void ModelLoader::register_tensor_with_paging(
    const std::string& name,
    uint32_t layer_idx,
    uint32_t tensor_type,
    uint64_t file_offset,
    uint64_t size_bytes,
    DType dtype)
{
    uint64_t page_id = paging_mgr_.register_tensor(
        name, layer_idx, tensor_type, 
        file_offset, size_bytes, dtype);
    
    tensor_pages_[name] = page_id;
    
    // Also store in layer structure if it's a layer tensor
    if (layer_idx < config_.n_layer && layer_idx != 0xFFFFFFFF) {
        if (layer_idx >= layer_tensors_.size()) {
            layer_tensors_.resize(layer_idx + 1);
        }
        
        auto& lt = layer_tensors_[layer_idx];
        if (name.find("q_weight") != std::string::npos) lt.attn_q = page_id;
        else if (name.find("k_weight") != std::string::npos) lt.attn_k = page_id;
        else if (name.find("v_weight") != std::string::npos) lt.attn_v = page_id;
        else if (name.find("o_weight") != std::string::npos) lt.attn_o = page_id;
        else if (name.find("gate_weight") != std::string::npos) lt.mlp_gate = page_id;
        else if (name.find("up_weight") != std::string::npos) lt.mlp_up = page_id;
        else if (name.find("down_weight") != std::string::npos) lt.mlp_down = page_id;
        else if (name.find("attn_norm") != std::string::npos) lt.attn_norm = page_id;
        else if (name.find("ffn_norm") != std::string::npos) lt.ffn_norm = page_id;
    }
}

std::optional<uint64_t> ModelLoader::get_tensor_page(const std::string& canonical_name) const {
    auto it = tensor_pages_.find(canonical_name);
    if (it != tensor_pages_.end()) return it->second;
    return std::nullopt;
}

std::optional<ModelLoader::LayerTensors> ModelLoader::get_layer_tensors(uint32_t layer_idx) const {
    if (layer_idx < layer_tensors_.size()) return layer_tensors_[layer_idx];
    return std::nullopt;
}

void ModelLoader::prefetch_embeddings() {
    // Embeddings are always needed, prefetch them first
    for (const auto& [name, page_id] : tensor_pages_) {
        if (name.find("token_embd") != std::string::npos ||
            name.find("lm_head") != std::string::npos) {
            paging_mgr_.prefetch_tensor(page_id, 20);  // Highest priority
        }
    }
}

void ModelLoader::prefetch_all_norms() {
    // All norm weights are tiny, load them all
    for (const auto& [name, page_id] : tensor_pages_) {
        if (name.find("_norm") != std::string::npos) {
            paging_mgr_.prefetch_tensor(page_id, 15);
        }
    }
}

} // namespace sl
