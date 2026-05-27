#pragma once
#include "common/platform.h"

#include "common/types.h"
#include "paging_manager.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace sl {

// ====================================================================
// GGUF File Format Reader
// 
// Reads GGUF v2/v3 model files and extracts:
// - Model hyperparameters (model config)
// - Tensor metadata (names, shapes, offsets, quantization)
// - Tokenizer information
// 
// GGUF format structure:
// [Magic Number (4 bytes)]
// [Version (4 bytes)]
// [Tensor count (8 bytes)]
// [Metadata key-value count (8 bytes)]
// [Metadata K-V pairs]
// [Tensor infos]
// [Alignment padding]
// [Tensor data (raw, interleaved)]
// ====================================================================
class GGUFReader {
public:
    explicit GGUFReader(const std::string& filepath);
    ~GGUFReader();
    
    bool is_valid() const { return valid_; }
    
    // Read model configuration (hyperparameters)
    ModelConfig read_config();
    
    // Enumerate all tensors with their metadata
    struct TensorInfo {
        std::string name;
        TensorShape shape;
        DType dtype;
        uint64_t offset;        // Byte offset in file
        uint64_t size_bytes;
    };
    std::vector<TensorInfo> enumerate_tensors();
    
    // Read raw tensor data at given offset
    void read_tensor_data(uint64_t offset, void* buffer, uint64_t size);
    
    // Memory-map a tensor region
    void* map_tensor_region(uint64_t offset, uint64_t size);
    
    uint64_t file_size() const;
    const std::string& filepath() const { return filepath_; }
    
private:
    std::string filepath_;
    HANDLE file_handle_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_handle_ = nullptr;
    void* mapped_base_ = nullptr;
    uint64_t file_size_ = 0;
    bool valid_ = false;
    
    // Parsed metadata
    uint32_t gguf_version_ = 0;
    uint64_t tensor_count_ = 0;
    uint64_t metadata_count_ = 0;
    std::unordered_map<std::string, std::string> metadata_;
    
    struct GGUFType {
        enum : uint32_t {
            U8 = 0, I8 = 1, U16 = 2, I16 = 3,
            U32 = 4, I32 = 5, F32 = 6, BOOL = 7,
            STRING = 8, ARRAY = 9, U64 = 10, I64 = 11,
            F64 = 12
        };
    };
    
    bool parse_header();
    std::string read_string_at(uint64_t& offset);
    
    // Layout reconstruction
    // Map tensor names to layers (heuristic from GGUF naming convention)
    // e.g., "blk.0.attn_q.weight" → layer=0, type=attention, tensor=q_weight
public:
public:
    void classify_tensor(const std::string& name, uint32_t& layer_idx, 
                         uint32_t& tensor_type, std::string& canonical_name);
};

// ====================================================================
// ModelLoader - high-level model loading orchestrator
// 
// Coordinates:
// 1. GGUF parsing
// 2. Paging system registration
// 3. Memory budget allocation
// 4. Prefetch scheduling
// ====================================================================
class ModelLoader {
public:
    ModelLoader(SSDPagingManager& paging_mgr);
    ~ModelLoader();
    
    // Load a model from GGUF file
    // Returns model configuration and populates the paging system
    ModelConfig load(const std::string& model_path);
    
    // Unload current model
    void unload();
    
    // Get tensor page IDs by name
    std::optional<uint64_t> get_tensor_page(const std::string& canonical_name) const;
    
    // Get tensor page IDs for a specific layer
    struct LayerTensors {
        uint64_t attn_q, attn_k, attn_v, attn_o;
        uint64_t mlp_gate, mlp_up, mlp_down;
        uint64_t attn_norm, ffn_norm;
    };
    std::optional<LayerTensors> get_layer_tensors(uint32_t layer_idx) const;
    
    // Prefetch strategy: load all norm + embedding weights first (small),
    // then stream layers on demand
    void prefetch_embeddings();
    void prefetch_all_norms();
    
    const ModelConfig& config() const { return config_; }
    bool is_loaded() const { return loaded_; }
    
private:
    SSDPagingManager& paging_mgr_;
    ModelConfig config_;
    bool loaded_ = false;
    
    // Tensor page ID registry
    std::unordered_map<std::string, uint64_t> tensor_pages_;
    
    // Layer-to-tensors mapping
    std::vector<LayerTensors> layer_tensors_;
    
    void register_tensor_with_paging(const std::string& name, 
                                      uint32_t layer_idx,
                                      uint32_t tensor_type,
                                      uint64_t file_offset,
                                      uint64_t size_bytes,
                                      DType dtype);
};

} // namespace sl