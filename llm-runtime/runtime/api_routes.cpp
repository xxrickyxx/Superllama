#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif
#include "api_routes.h"
#include "engine/inference.h"
#include "models/model_registry.h"
#include "models/downloader.h"
#include "common/logging.h"

namespace sl {

APIRouter::APIRouter() {
    // Register all Ollama-compatible API routes
    register_route("POST", "/api/generate", api_handlers::generate);
    register_route("POST", "/api/chat", api_handlers::chat);
    register_route("GET", "/api/tags", api_handlers::tags);
    register_route("POST", "/api/pull", api_handlers::pull);
    register_route("DELETE", "/api/delete", api_handlers::delete_model);
    register_route("POST", "/api/show", api_handlers::show);
    register_route("POST", "/api/embeddings", api_handlers::embeddings);
    register_route("GET", "/api/version", api_handlers::version);
}

void APIRouter::register_route(const std::string& method, const std::string& path, 
                                RouteHandler handler) {
    routes_.push_back({method, path, handler});
}

std::string APIRouter::dispatch(const std::string& method, const std::string& path,
                                 const std::string& body, RouteContext ctx,
                                 bool is_streaming, StreamCallback stream) {
    json req_json;
    try {
        if (!body.empty()) req_json = json::parse(body);
    } catch (...) {
        return R"({"error":"Invalid JSON"})";
    }
    
    for (const auto& route : routes_) {
        if (route.method == method && route.path == path) {
            try {
                return route.handler(req_json, ctx, stream);
        } catch (const std::exception& e) {
            json err;
            err["error"] = std::string("Handler error: ") + e.what();
            return err.dump();
        }
        }
    }
    
    json err;
    err["error"] = std::string("Route not found: ") + method + " " + path;
    return err.dump();
}

namespace api_handlers {

std::string generate(const json& req_json, RouteContext ctx, StreamCallback stream) {
    if (!ctx.engine) return R"({"error":"No inference engine"})";
    
    InferenceRequest req;
    req.model_name = req_json.value("model", "");
    req.prompt = req_json.value("prompt", "");
    req.chat_mode = false;
    
    auto& gc = req.gen_config;
    gc.stream = req_json.value("stream", true);
    gc.max_tokens = req_json.value("max_tokens", 256);
    gc.temperature = req_json.value("temperature", 0.7f);
    gc.top_p = req_json.value("top_p", 0.9f);
    gc.top_k = req_json.value("top_k", 40);
    gc.repetition_penalty = req_json.value("repeat_penalty", 1.1f);
    
    if (req_json.contains("system")) {
        req.system_prompt = req_json["system"].get<std::string>();
    }
    
    if (req_json.contains("options")) {
        auto opts = req_json["options"];
        if (opts.contains("temperature")) gc.temperature = opts["temperature"];
        if (opts.contains("top_p")) gc.top_p = opts["top_p"];
        if (opts.contains("top_k")) gc.top_k = opts["top_k"];
        if (opts.contains("num_predict")) gc.max_tokens = opts["num_predict"];
    }
    
    std::string response_text;
    auto tokens = ctx.engine->generate(req, [&](const InferenceToken& token) {
        if (stream && !token.text.empty()) {
            response_text += token.text;
            json chunk;
            chunk["model"] = req.model_name;
            chunk["response"] = token.text;
            chunk["done"] = token.is_final;
            stream(chunk.dump());
        } else {
            response_text += token.text;
        }
    });
    
    json final_response;
    final_response["model"] = req.model_name;
    final_response["response"] = response_text;
    final_response["done"] = true;
    
    if (req_json.contains("context")) {
        final_response["context"] = json::array();  // token IDs would go here
    }
    
    return final_response.dump();
}

std::string chat(const json& req_json, RouteContext ctx, StreamCallback stream) {
    if (!ctx.engine) return R"({"error":"No inference engine"})";
    
    InferenceRequest req;
    req.model_name = req_json.value("model", "");
    req.chat_mode = true;
    
    auto& gc = req.gen_config;
    gc.stream = req_json.value("stream", true);
    gc.max_tokens = req_json.value("max_tokens", 256);
    gc.temperature = req_json.value("temperature", 0.7f);
    gc.top_p = req_json.value("top_p", 0.9f);
    
    for (const auto& msg : req_json["messages"]) {
        InferenceRequest::Message m;
        m.role = msg.value("role", "user");
        m.content = msg.value("content", "");
        req.messages.push_back(m);
    }
    
    if (req_json.contains("options")) {
        auto opts = req_json["options"];
        if (opts.contains("temperature")) gc.temperature = opts["temperature"];
    }
    
    std::string response_content;
    auto tokens = ctx.engine->generate(req, [&](const InferenceToken& token) {
        if (stream && !token.text.empty()) {
            response_content += token.text;
            json chunk;
            chunk["model"] = req.model_name;
            chunk["message"]["role"] = "assistant";
            chunk["message"]["content"] = token.text;
            chunk["done"] = token.is_final;
            stream(chunk.dump());
        } else {
            response_content += token.text;
        }
    });
    
    json final_response;
    final_response["model"] = req.model_name;
    final_response["message"]["role"] = "assistant";
    final_response["message"]["content"] = response_content;
    final_response["done"] = true;
    
    return final_response.dump();
}

std::string tags(const json& req_json, RouteContext ctx, StreamCallback) {
    if (!ctx.registry) return R"({"models":[]})";
    
    json response;
    response["models"] = json::array();
    
    auto models = ctx.registry->list_models();
    for (const auto& m : models) {
        json model_json;
        model_json["name"] = m.name;
        model_json["modified_at"] = 
            std::chrono::duration_cast<std::chrono::seconds>(
                m.modified_at.time_since_epoch()).count();
        model_json["size"] = m.size_bytes;
        model_json["digest"] = m.digest;
        
        json details;
        details["format"] = dtype_name(m.quantization);
        details["family"] = m.family;
        details["families"] = json::array({m.family});
        char param_buf[32];
        snprintf(param_buf, sizeof(param_buf), "%.1fB", m.param_count / 1e9);
        details["parameter_size"] = std::string(param_buf);
        details["quantization_level"] = dtype_name(m.quantization);
        
        model_json["details"] = details;
        response["models"].push_back(model_json);
    }
    
    return response.dump();
}

std::string pull(const json& req_json, RouteContext ctx, StreamCallback stream) {
    std::string model_name = req_json.value("model", "");
    bool insecure = req_json.value("insecure", false);
    
    if (model_name.empty()) {
        json err;
        err["error"] = "model name is required";
        return err.dump();
    }
    
    SL_LOG_INFO("Pull request for model: {}", model_name);
    
    // Check if model already exists
    if (ctx.registry) {
        auto existing = ctx.registry->find_by_name(model_name);
        if (existing.has_value()) {
            json already;
            already["status"] = "already exists";
            already["message"] = std::string("Model '") + model_name + "' is already installed";
            return already.dump();
        }
    }
    
    // Use the downloader if available
    if (ctx.downloader) {
        // Build download URL: first try Ollama registry format (org/model:tag)
        // If it looks like a simple filename, assume it's from a known registry
        std::string download_url;
        std::string dest_filename;
        
        // Determine destination path
        std::string models_dir = "./models";
        std::filesystem::path dest_dir(models_dir);
        if (!std::filesystem::exists(dest_dir)) {
            std::filesystem::create_directories(dest_dir);
        }
        
        // If model_name contains "/" it's a full registry reference
        // Supported formats:
        //   "org/repo/filename.gguf"  -> HuggingFace direct file
        //   "org/repo:tag"            -> HuggingFace repo with branch tag
        //   "org/repo"                -> HuggingFace repo (use main branch)
        if (model_name.find('/') != std::string::npos) {
            auto last_slash = model_name.rfind('/');
            std::string last_part = model_name.substr(last_slash + 1);

            if (last_part.find(".gguf") != std::string::npos) {
                // Format: "org/repo/filename.gguf" — split at last slash
                std::string repo_path = model_name.substr(0, last_slash);
                dest_filename = last_part;
                download_url = "https://huggingface.co/" + repo_path + "/resolve/main/" + dest_filename;
            } else {
                // Format: "org/repo" or "org/repo:tag"
                std::string org_model = model_name;
                std::string tag = "main";
                auto colon_pos = org_model.find(':');
                if (colon_pos != std::string::npos) {
                    tag = org_model.substr(colon_pos + 1);
                    org_model = org_model.substr(0, colon_pos);
                }

                dest_filename = org_model;
                for (auto& c : dest_filename) {
                    if (c == '/') c = '-';
                }
                dest_filename += ".gguf";
                download_url = "https://huggingface.co/" + org_model + "/resolve/" + tag + "/" + dest_filename;
            }
        } else {
            // Simple model name like "Llama-3.3-70B-Instruct-Q4_K_M.gguf"
            // Try Ollama registry first
            dest_filename = model_name;
            if (dest_filename.find(".gguf") == std::string::npos) {
                dest_filename += ".gguf";
            }
            
            // Map well-known models to their URLs
            // This is a basic resolver; in production this would query a registry API
            if (model_name.find("Llama-3.3-70B") != std::string::npos ||
                model_name.find("llama-3.3-70b") != std::string::npos) {
                download_url = "https://huggingface.co/bartowski/Llama-3.3-70B-Instruct-GGUF/resolve/main/" + dest_filename;
            } else if (model_name.find("Llama-3.2-3B") != std::string::npos ||
                       model_name.find("llama-3.2-3b") != std::string::npos) {
                download_url = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/" + dest_filename;
            } else if (model_name.find("Llama-3.2-1B") != std::string::npos ||
                       model_name.find("llama-3.2-1b") != std::string::npos) {
                download_url = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/" + dest_filename;
            } else if (model_name.find("Llama-3.1-8B") != std::string::npos ||
                       model_name.find("llama-3.1-8b") != std::string::npos) {
                download_url = "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/" + dest_filename;
            } else if (model_name.find("Mistral-7B") != std::string::npos ||
                       model_name.find("mistral-7b") != std::string::npos) {
                download_url = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/" + dest_filename;
            } else if (model_name.find("Qwen2.5") != std::string::npos ||
                       model_name.find("qwen2.5") != std::string::npos) {
                download_url = "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/" + dest_filename;
            } else {
                // Generic: try HuggingFace bartowski org as default
                // Extract base model name
                std::string base = dest_filename;
                auto dot_pos = base.find(".gguf");
                if (dot_pos != std::string::npos) {
                    base = base.substr(0, dot_pos);
                }
                // Default to bartowski org
                download_url = "https://huggingface.co/bartowski/" + base + "-GGUF/resolve/main/" + dest_filename;
            }
        }
        
        std::string dest_path = (dest_dir / dest_filename).string();
        
        SL_LOG_INFO("Downloading model: {} from {} to {}", model_name, download_url, dest_path);
        
        // Initiate download (non-blocking; the server will stream progress)
        bool started = ctx.downloader->download(
            model_name, download_url, dest_path,
            [model_name, stream](const DownloadProgress& progress) {
                json status;
                status["model"] = model_name;
                status["status"] = progress.status;
                status["total"] = progress.total_bytes;
                status["completed"] = progress.downloaded_bytes;
                
                if (!progress.error_message.empty()) {
                    status["error"] = progress.error_message;
                }
                
                if (progress.completed) {
                    status["status"] = "success";
                }
                
                stream(status.dump());
            }
        );
        
        if (started) {
            json status;
            status["status"] = "initiated";
            status["message"] = std::string("Download started for '") + model_name + "' from " + download_url;
            return status.dump();
        } else {
            json err;
            err["error"] = std::string("Failed to start download for '") + model_name + "' (may already be in progress)";
            return err.dump();
        }
    }
    
    // No downloader available
    json status;
    status["status"] = "error";
    status["error"] = std::string("Download infrastructure not available for model: ") + model_name;
    return status.dump();
}

std::string delete_model(const json& req_json, RouteContext ctx, StreamCallback) {
    std::string model_name = req_json.value("model", "");
    
    if (ctx.registry) {
        ctx.registry->delete_model(model_name);
    }
    
    return "";
}

std::string show(const json& req_json, RouteContext ctx, StreamCallback) {
    std::string model_name = req_json.value("model", "");
    
    json response;
    
    if (ctx.registry) {
        auto opt_model = ctx.registry->find_by_name(model_name);
        if (opt_model) {
            response["license"] = "LLAMA 3 COMMUNITY LICENSE";
            response["modelfile"] = "# Model file information";
            
            json params;
            params["format"] = dtype_name(opt_model->quantization);
            response["parameters"] = params.dump();
        }
    }
    
    return response.dump();
}

std::string embeddings(const json& req_json, RouteContext ctx, StreamCallback) {
    if (!ctx.engine) return R"({"error":"No inference engine"})";
    
    std::string prompt = req_json.value("prompt", "");
    // Would run the model up to the last layer and return the hidden state
    // as embeddings vector
    
    json response;
    response["embedding"] = json::array();  // placeholder
    return response.dump();
}

std::string version(const json&, RouteContext, StreamCallback) {
    json response;
    response["version"] = "0.1.0";
    return response.dump();
}

} // namespace api_handlers
} // namespace sl