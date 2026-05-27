#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif
#include "inference.h"
#include "common/logging.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <functional>
#include <functional>
#include <sstream>

namespace sl {

InferenceEngine::InferenceEngine() {
    // Logger setup
}

InferenceEngine::~InferenceEngine() {
    cancel();
    if (generation_thread_.joinable()) {
        generation_thread_.join();
    }
}

bool InferenceEngine::initialize(ModelLoader& loader) {
    loader_ = &loader;
    config_ = loader.config();
    
    // Load embedding weights (always resident - small)
    load_embedding_weights(loader);
    
    // Create transformer blocks
    create_transformer_blocks();
    
    // Allocate KV cache
    kv_cache_ = std::make_unique<KVCacheManager>(
        config_.n_layer,
        config_.n_kv_head(),
        config_.head_dim(),
        config_.n_ctx
    );
    
    // Allocate hidden state buffer
    hidden_state_.resize(config_.n_embd);
    logits_.resize(config_.n_vocab);
    
    SL_LOG_INFO("InferenceEngine initialized: {} layers, {:.0f} MB KV cache",
                config_.n_layer, kv_cache_->total_allocated_bytes() / 1048576.0);
    
    return true;
}

void InferenceEngine::load_embedding_weights(ModelLoader& loader) {
    // These are small enough to keep resident
    token_embeddings_ = Tensor({config_.n_vocab, config_.n_embd}, DType::F32);
    lm_head_ = Tensor({config_.n_embd, config_.n_vocab}, DType::F32);
    final_norm_ = Tensor({config_.n_embd}, DType::F32);
    
    // In production, resolve from paging system:
    // auto tok_embd_page = loader.get_tensor_page("token_embd");
    // if (tok_embd_page) {
    //     void* data = paging_mgr_->resolve_tensor(*tok_embd_page);
    //     memcpy(token_embeddings_.data(), data, token_embeddings_.nbytes());
    // }
}

void InferenceEngine::create_transformer_blocks() {
    blocks_.reserve(config_.n_layer);
    for (uint32_t i = 0; i < config_.n_layer; i++) {
        blocks_.push_back(std::make_unique<TransformerBlock>(config_, i));
    }
}

void InferenceEngine::run_transformer_layer(uint32_t layer_idx) {
    if (layer_idx >= blocks_.size()) return;
    
    // Notify paging system which layer is active
    if (paging_mgr_) {
        paging_mgr_->set_active_layer(layer_idx);
    }
    
    // Get KV cache for this layer
    KVCacheSlot* layer_kv = kv_cache_->get_layer_cache(layer_idx);
    
    // Get current token position
    uint32_t position = kv_cache_->used_entries();
    
    // Run the transformer block
    blocks_[layer_idx]->forward(hidden_state_.data(), layer_kv, position);
}

void InferenceEngine::compute_logits() {
    // Apply final RMS norm
    simd::rms_norm_f32(
        hidden_state_.data(), 
        hidden_state_.data(),
        config_.n_embd, 
        config_.f_norm_eps
    );
    
    // Apply final norm weights
    const float* norm_w = static_cast<const float*>(final_norm_.data());
    for (size_t i = 0; i < config_.n_embd; i++) {
        hidden_state_[i] *= norm_w[i];
    }
    
    // Project to vocab: logits = hidden_state @ lm_head
    simd::matvec_f32(
        static_cast<const float*>(lm_head_.data()),
        hidden_state_.data(),
        logits_.data(),
        config_.n_vocab,
        config_.n_embd
    );
}

Token InferenceEngine::sample_token(const GenerationConfig& gen_cfg) {
    // Apply temperature
    if (gen_cfg.temperature > 0.0f) {
        float inv_temp = 1.0f / gen_cfg.temperature;
        for (auto& l : logits_) l *= inv_temp;
    }
    
    // Apply softmax
    simd::softmax_f32(logits_.data(), logits_.data(), config_.n_vocab);
    
    // Top-K filtering
    if (gen_cfg.top_k > 0 && gen_cfg.top_k < config_.n_vocab) {
        // Find top-k threshold
        std::vector<std::pair<float, Token>> sorted;
        sorted.reserve(config_.n_vocab);
        for (Token i = 0; i < static_cast<Token>(config_.n_vocab); i++) {
            sorted.emplace_back(logits_[i], i);
        }
        std::partial_sort(sorted.begin(), 
                          sorted.begin() + gen_cfg.top_k, 
                          sorted.end(),
                          std::greater<>());
        
        float threshold = sorted[gen_cfg.top_k - 1].first;
        for (auto& l : logits_) {
            if (l < threshold) l = 0.0f;
        }
    }
    
    // Top-P (nucleus) sampling
    if (gen_cfg.top_p < 1.0f) {
        std::vector<std::pair<float, Token>> sorted;
        sorted.reserve(config_.n_vocab);
        for (Token i = 0; i < static_cast<Token>(config_.n_vocab); i++) {
            if (logits_[i] > 0) sorted.emplace_back(logits_[i], i);
        }
        std::sort(sorted.begin(), sorted.end(), std::greater<>());
        
        float cumsum = 0.0f;
        size_t cutoff = 0;
        for (size_t i = 0; i < sorted.size(); i++) {
            cumsum += sorted[i].first;
            if (cumsum >= gen_cfg.top_p) {
                cutoff = i + 1;
                break;
            }
        }
        
        // Zero out tokens beyond cutoff
        if (cutoff < sorted.size()) {
            for (size_t i = cutoff; i < sorted.size(); i++) {
                logits_[sorted[i].second] = 0.0f;
            }
        }
    }
    
    // Renormalize
    float sum = 0.0f;
    for (auto& l : logits_) sum += l;
    if (sum > 0) {
        for (auto& l : logits_) l /= sum;
    }
    
    // Sample
    static thread_local std::mt19937_64 rng(
        std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    
    float cumsum = 0.0f;
    for (Token i = 0; i < static_cast<Token>(config_.n_vocab); i++) {
        cumsum += logits_[i];
        if (cumsum >= r) {
            return i;
        }
    }
    
    // Fallback: argmax
    return static_cast<Token>(
        std::max_element(logits_.begin(), logits_.end()) - logits_.begin());
}

std::vector<Token> InferenceEngine::tokenize(const std::string& text) {
    // Placeholder: in production, this would use SentencePiece or BPE
    // For now, we just return dummy tokens
    std::vector<Token> tokens;
    tokens.push_back(config_.bos_token_id);
    
    // Split by spaces (extremely simplified)
    std::istringstream iss(text);
    std::string word;
    while (iss >> word) {
        // Map each "word" to a token ID (placeholder)
        tokens.push_back(static_cast<Token>(std::hash<std::string>{}(word) % config_.n_vocab));
    }
    
    return tokens;
}

std::string InferenceEngine::detokenize(Token token_id) {
    // Placeholder
    if (token_id == config_.eos_token_id) return "";
    if (token_id == config_.bos_token_id) return "";
    
    // In production, look up in vocabulary
    return std::format("<token_{}>", token_id);
}

std::vector<InferenceToken> InferenceEngine::generate(
    const InferenceRequest& request,
    TokenCallback callback)
{
    std::vector<InferenceToken> results;
    
    if (generating_) {
        SL_LOG_WARN("Generation already in progress");
        return results;
    }
    
    generating_ = true;
    cancel_requested_ = false;
    
    // Build prompt from messages or direct prompt
    std::string full_prompt;
    if (request.chat_mode && !request.messages.empty()) {
        for (const auto& msg : request.messages) {
            if (msg.role == "system") {
                full_prompt += "<|system|>\n" + msg.content + "\n";
            } else if (msg.role == "user") {
                full_prompt += "<|user|>\n" + msg.content + "\n";
            } else if (msg.role == "assistant") {
                full_prompt += "<|assistant|>\n" + msg.content + "\n";
            }
        }
        full_prompt += "<|assistant|>\n";
    } else {
        full_prompt = request.prompt;
    }
    
    // Tokenize
    auto input_tokens = tokenize(full_prompt);
    generated_tokens_ = input_tokens;
    
    // Prefill: process all input tokens (batch forward for context)
    kv_cache_->clear();
    
    SL_LOG_INFO("Starting generation with {} input tokens, max {} new tokens",
                input_tokens.size(), request.gen_config.max_tokens);
    
    auto start_time = std::chrono::steady_clock::now();
    
    for (size_t pos = 0; pos < input_tokens.size(); pos++) {
        // Get embedding for this token
        uint32_t token_id = input_tokens[pos];
        
        // Look up embedding
        if (token_id < config_.n_vocab) {
            const float* emb = static_cast<const float*>(token_embeddings_.data()) 
                               + token_id * config_.n_embd;
            memcpy(hidden_state_.data(), emb, config_.n_embd * sizeof(float));
        } else {
            std::fill(hidden_state_.begin(), hidden_state_.end(), 0.0f);
        }
        
        // Run all transformer layers (prefill)
        for (uint32_t l = 0; l < config_.n_layer; l++) {
            run_transformer_layer(l);
        }
    }
    
    // Autoregressive generation loop
    for (uint32_t gen_step = 0; gen_step < request.gen_config.max_tokens; gen_step++) {
        if (cancel_requested_) {
            SL_LOG_INFO("Generation cancelled at step {}", gen_step);
            break;
        }
        
        // Compute logits from final hidden state
        compute_logits();
        
        // Sample next token
        Token next_token = sample_token(request.gen_config);
        
        // Check stop conditions
        if (next_token == config_.eos_token_id) {
            InferenceToken tok{next_token, "", 0.0f, true};
            results.push_back(tok);
            if (callback) callback(tok);
            break;
        }
        
        // Check stop tokens from config
        bool stopped = false;
        for (auto stop_tok : request.gen_config.stop_tokens) {
            if (next_token == stop_tok) { stopped = true; break; }
        }
        if (stopped) break;
        
        // Detokenize
        std::string text = detokenize(next_token);
        
        // Check stop strings
        for (const auto& stop_str : request.gen_config.stop_strings) {
            if (text.find(stop_str) != std::string::npos) {
                stopped = true;
                break;
            }
        }
        
        // Create result token
        float log_prob = std::log(std::max(logits_[next_token], 1e-10f));
        InferenceToken result_tok{next_token, text, log_prob, false};
        
        results.push_back(result_tok);
        generated_tokens_.push_back(next_token);
        
        // Streaming callback
        if (callback) {
            callback(result_tok);
        }
        
        if (stopped) break;
        
        // Prepare for next iteration: use the new token's embedding
        if (next_token < config_.n_vocab) {
            const float* emb = static_cast<const float*>(token_embeddings_.data()) 
                               + next_token * config_.n_embd;
            memcpy(hidden_state_.data(), emb, config_.n_embd * sizeof(float));
        }
        
        // Run transformer on this single token
        for (uint32_t l = 0; l < config_.n_layer; l++) {
            run_transformer_layer(l);
        }
    }
    
    auto end_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    
    uint32_t new_tokens = static_cast<uint32_t>(results.size());
    double tps = elapsed > 0 ? (new_tokens * 1000.0 / elapsed) : 0.0;
    
    SL_LOG_INFO("Generation complete: {} tokens in {}ms ({:.1f} tok/s)",
                new_tokens, elapsed, tps);
    
    // Mark final token
    if (!results.empty()) {
        results.back().is_final = true;
        if (callback) callback(results.back());
    }
    
    generating_ = false;
    return results;
}

void InferenceEngine::cancel() {
    cancel_requested_ = true;
}

PagingStats InferenceEngine::get_paging_stats() const {
    if (paging_mgr_) {
        return paging_mgr_->get_stats();
    }
    return PagingStats{};
}

void InferenceEngine::warmup() {
    // Prefetch first few layers so inference starts fast
    if (paging_mgr_) {
        paging_mgr_->prefetch_layer(0);
        paging_mgr_->prefetch_layer(1);
        paging_mgr_->prefetch_layer(2);
    }
    
    // Run a dummy forward pass to warm CPU cache and JIT
    if (blocks_.size() > 0) {
        std::vector<float> dummy_hidden(config_.n_embd, 0.0f);
        KVCacheSlot* dummy_kv = kv_cache_->get_layer_cache(0);
        blocks_[0]->forward(dummy_hidden.data(), dummy_kv, 0);
    }
}

// Simple hash for tokenizer placeholder
static inline uint64_t hash(const std::string& s) {
    uint64_t h = 14695981039346656037ULL;
    for (char c : s) h = (h ^ static_cast<uint8_t>(c)) * 1099511628211ULL;
    return h;
}

// ====================================================================
// NVMeSSDBackend (stub)
// ====================================================================
bool NVMeSSDBackend::open(const std::string& path) {
    file_handle_ = CreateFileA(
        path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
        OPEN_EXISTING,
        FILE_FLAG_OVERLAPPED | FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN,
        nullptr);
    
    if (file_handle_ == INVALID_HANDLE_VALUE) return false;
    
    LARGE_INTEGER li;
    GetFileSizeEx(file_handle_, &li);
    file_size_ = li.QuadPart;
    
    mapping_handle_ = CreateFileMappingA(file_handle_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapping_handle_) {
        mapped_base_ = MapViewOfFile(mapping_handle_, FILE_MAP_READ, 0, 0, 0);
    }
    
    iocp_ = CreateIoCompletionPort(INVALID_HANDLE_VALUE, nullptr, 0, 0);
    CreateIoCompletionPort(file_handle_, iocp_, 0, 0);
    
    return true;
}

void NVMeSSDBackend::close() {
    if (mapped_base_) { UnmapViewOfFile(mapped_base_); mapped_base_ = nullptr; }
    if (mapping_handle_) { CloseHandle(mapping_handle_); mapping_handle_ = nullptr; }
    if (iocp_) { CloseHandle(iocp_); iocp_ = nullptr; }
    if (file_handle_ != INVALID_HANDLE_VALUE) { CloseHandle(file_handle_); file_handle_ = INVALID_HANDLE_VALUE; }
}

void NVMeSSDBackend::read(uint64_t offset, void* buffer, uint64_t size) {
    if (mapped_base_) {
        memcpy(buffer, static_cast<uint8_t*>(mapped_base_) + offset, size);
    }
}

void NVMeSSDBackend::async_read(uint64_t offset, void* buffer, uint64_t size,
                                 std::function<void(bool)> callback) {
    // Simplified sync implementation
    read(offset, buffer, size);
    if (callback) callback(true);
}

void* NVMeSSDBackend::map(uint64_t offset, uint64_t size) {
    if (mapped_base_) {
        return static_cast<uint8_t*>(mapped_base_) + offset;
    }
    return nullptr;
}

uint64_t NVMeSSDBackend::file_size() const { return file_size_; }

} // namespace sl
