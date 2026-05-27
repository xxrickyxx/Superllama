#pragma once
#include "common/platform.h"

#include "common/types.h"
#include "engine/inference.h"
#include "models/model_registry.h"
#include "models/downloader.h"
#include <string>
#include <thread>
#include <atomic>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <queue>

#ifdef _WIN32
#include <winsock2.h>
#pragma comment(lib, "ws2_32.lib")
#endif

namespace sl {

// ====================================================================
// HTTPServer - lightweight embedded HTTP server for Ollama-compatible API
// 
// Uses Windows HTTP Server API (HTTPAPI.dll) for production performance,
// or falls back to a simple socket-based server.
// ====================================================================
class HTTPServer {
public:
    HTTPServer(const ServerConfig& config);
    ~HTTPServer();
    
    // Start the server (non-blocking)
    bool start();
    
    // Stop the server
    void stop();
    
    // Wait for server to finish
    void wait();
    
    // Check if server is running
    bool is_running() const { return running_; }
    
    // Set the inference engine (for API handlers)
    void set_inference_engine(InferenceEngine* engine) { inference_ = engine; }
    
    // Set the model registry (for API handlers)
    void set_model_registry(ModelRegistry* registry) { registry_ = registry; }
    
    // Set the model downloader (for API handlers)
    void set_model_downloader(ModelDownloader* downloader) { downloader_ = downloader; }
    
    // Set the UI directory for static file serving
    void set_ui_directory(const std::string& dir) { ui_dir_ = dir; }
    
    // Port number
    uint16_t port() const { return port_; }
    
private:
    std::string host_;
    uint16_t port_;
    ServerConfig config_;
    
    InferenceEngine* inference_ = nullptr;
    ModelRegistry* registry_ = nullptr;
    ModelDownloader* downloader_ = nullptr;
    
    std::atomic<bool> running_{false};
    std::thread server_thread_;
    
    // Windows HTTP Server API
    HANDLE request_queue_ = nullptr;
    HANDLE http_server_ = nullptr;
    
    // Socket fallback
    SOCKET server_socket_ = INVALID_SOCKET;
    std::thread accept_thread_;
    
    // Static file serving
    std::string ui_dir_;
    bool try_serve_static(const std::string& path, std::string& content_type, 
                          std::string& response_body, uint64_t& content_length);
    std::string guess_mime_type(const std::string& path);
    
    void server_loop();
    void socket_server_loop();
    void handle_request(const std::string& method, const std::string& path,
                        const std::string& body, std::string& response,
                        bool& is_streaming,
                        std::function<void(const std::string&)> stream_callback);
    
    // API handlers
    std::string handle_generate(const std::string& body,
                                 std::function<void(const std::string&)> stream);
    std::string handle_chat(const std::string& body,
                             std::function<void(const std::string&)> stream);
    std::string handle_tags(const std::string& body);
    std::string handle_pull(const std::string& body,
                             std::function<void(const std::string&)> stream);
    std::string handle_delete(const std::string& body);
    
    // JSON helpers
    std::string build_generate_response(const InferenceToken& token, bool include_context);
    std::string build_chat_response(const InferenceToken& token, bool include_context);
    
    // In-flight generation tracking
    struct ActiveGeneration {
        std::string model_name;
        std::string id;
        std::chrono::steady_clock::time_point started;
        bool completed = false;
    };
    std::mutex active_gens_mutex_;
    std::vector<ActiveGeneration> active_generations_;
    uint64_t next_gen_id_ = 0;
};

} // namespace sl