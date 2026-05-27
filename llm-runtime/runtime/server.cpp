#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif
#include "server.h"
#include "api_routes.h"
#include "common/logging.h"
#include <nlohmann/json.hpp>
#include <winhttp.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <sstream>
#include <fstream>
#include <filesystem>

#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "winhttp.lib")

namespace sl {

using json = nlohmann::json;

// ====================================================================
// Static file serving
// ====================================================================
std::string HTTPServer::guess_mime_type(const std::string& path) {
    std::string ext;
    size_t dot = path.rfind('.');
    if (dot != std::string::npos) ext = path.substr(dot);
    
    if (ext == ".html" || ext == ".htm")   return "text/html; charset=utf-8";
    if (ext == ".css")                      return "text/css; charset=utf-8";
    if (ext == ".js")                       return "application/javascript; charset=utf-8";
    if (ext == ".json")                     return "application/json";
    if (ext == ".png")                      return "image/png";
    if (ext == ".jpg" || ext == ".jpeg")    return "image/jpeg";
    if (ext == ".svg")                      return "image/svg+xml";
    if (ext == ".ico")                      return "image/x-icon";
    if (ext == ".woff2")                    return "font/woff2";
    if (ext == ".woff")                     return "font/woff";
    if (ext == ".ttf")                      return "font/ttf";
    return "application/octet-stream";
}

bool HTTPServer::try_serve_static(const std::string& path, 
                                    std::string& content_type,
                                    std::string& response_body,
                                    uint64_t& content_length) {
    if (ui_dir_.empty()) return false;
    
    // Map / to /index.html, otherwise serve exact path
    std::string file_path = (path == "/" || path.empty()) ? "/index.html" : path;
    
    // Security: prevent directory traversal
    if (file_path.find("..") != std::string::npos) return false;
    
    std::string full_path = ui_dir_ + file_path;
    
    // Normalize separators for Windows
    for (auto& c : full_path) if (c == '/') c = '\\';
    
    if (!std::filesystem::exists(full_path) || 
        !std::filesystem::is_regular_file(full_path)) {
        return false;
    }
    
    // Read file
    std::ifstream file(full_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return false;
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    response_body.resize(static_cast<size_t>(size));
    if (!file.read(&response_body[0], size)) return false;
    
    content_type = guess_mime_type(file_path);
    content_length = static_cast<uint64_t>(size);
    return true;
}

// ====================================================================
// HTTPServer implementation
// ====================================================================

HTTPServer::HTTPServer(const ServerConfig& config)
    : host_(config.host)
    , port_(config.port)
    , config_(config) {}

HTTPServer::~HTTPServer() {
    stop();
}

bool HTTPServer::start() {
    if (running_) return true;
    
    running_ = true;
    
    // Try HTTP Server API first, fall back to socket
    server_thread_ = std::thread(&HTTPServer::socket_server_loop, this);
    
    SL_LOG_INFO("HTTP server starting on {}:{}", host_, port_);
    return true;
}

void HTTPServer::stop() {
    running_ = false;
    
    if (server_socket_ != INVALID_SOCKET) {
        closesocket(server_socket_);
        server_socket_ = INVALID_SOCKET;
    }
    
    if (accept_thread_.joinable()) {
        accept_thread_.join();
    }
    
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
    
    WSACleanup();
}

void HTTPServer::wait() {
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
}

void HTTPServer::socket_server_loop() {
    WSADATA wsa_data;
    if (WSAStartup(MAKEWORD(2, 2), &wsa_data) != 0) {
        SL_LOG_ERROR("WSAStartup failed");
        return;
    }
    
    server_socket_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (server_socket_ == INVALID_SOCKET) {
        SL_LOG_ERROR("Failed to create server socket");
        WSACleanup();
        return;
    }
    
    // Set socket options
    int opt = 1;
    setsockopt(server_socket_, SOL_SOCKET, SO_REUSEADDR, 
               reinterpret_cast<const char*>(&opt), sizeof(opt));
    
    sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port_);
    inet_pton(AF_INET, host_.c_str(), &addr.sin_addr);
    
    if (bind(server_socket_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR) {
        SL_LOG_ERROR("Bind failed: {}", WSAGetLastError());
        closesocket(server_socket_);
        server_socket_ = INVALID_SOCKET;
        WSACleanup();
        return;
    }
    
    if (listen(server_socket_, SOMAXCONN) == SOCKET_ERROR) {
        SL_LOG_ERROR("Listen failed: {}", WSAGetLastError());
        closesocket(server_socket_);
        server_socket_ = INVALID_SOCKET;
        WSACleanup();
        return;
    }
    
    SL_LOG_INFO("HTTP server listening on http://{}:{}/", host_, port_);
    
    while (running_) {
        fd_set read_set;
        FD_ZERO(&read_set);
        FD_SET(server_socket_, &read_set);
        
        timeval timeout;
        timeout.tv_sec = 1;
        timeout.tv_usec = 0;
        
        int result = select(0, &read_set, nullptr, nullptr, &timeout);
        
        if (result == SOCKET_ERROR || result == 0) continue;
        
        SOCKET client = accept(server_socket_, nullptr, nullptr);
        if (client == INVALID_SOCKET) continue;
        
        // Handle client in a thread
        std::thread([this, client]() {
            char buffer[65536];
            int recv_len = recv(client, buffer, sizeof(buffer) - 1, 0);
            
            if (recv_len > 0) {
                buffer[recv_len] = '\0';
                std::string request_str(buffer, recv_len);
                
                // Parse HTTP request
                std::string method, path, body;
                std::istringstream stream(request_str);
                std::string line;
                
                // First line: METHOD /path HTTP/1.1
                if (!std::getline(stream, line)) {
                    closesocket(client);
                    return;
                }
                
                {
                    std::istringstream first_line(line);
                    std::string http_ver;
                    first_line >> method >> path >> http_ver;
                }
                
                // Collect headers
                std::string headers;
                while (std::getline(stream, line)) {
                    // Trim \r
                    if (!line.empty() && line.back() == '\r') line.pop_back();
                    if (line.empty()) break;
                    headers += line + "\n";
                }
                
                // Body is everything after headers
                size_t header_end = request_str.find("\r\n\r\n");
                if (header_end != std::string::npos) {
                    body = request_str.substr(header_end + 4);
                }
                
                // CORS preflight: respond with proper headers
                if (method == "OPTIONS") {
                    const char* preflight =
                        "HTTP/1.1 204 No Content\r\n"
                        "Access-Control-Allow-Origin: *\r\n"
                        "Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\n"
                        "Access-Control-Allow-Headers: Content-Type, X-Stream\r\n"
                        "Access-Control-Max-Age: 86400\r\n"
                        "Connection: close\r\n\r\n";
                    send(client, preflight, static_cast<int>(strlen(preflight)), 0);
                    closesocket(client);
                    return;
                }
                
                // Detect streaming: either header-driven or path-driven
                bool is_streaming = false;
                bool sse_headers_sent = false;
                bool is_streaming_by_path = (path == "/api/generate" || path == "/api/generate/" ||
                                             path == "/api/chat" || path == "/api/chat/" ||
                                             path == "/api/pull" || path == "/api/pull/");
                
                if (headers.find("X-Stream: true") != std::string::npos ||
                    path.find("stream=true") != std::string::npos ||
                    is_streaming_by_path) {
                    is_streaming = true;
                }
                
                // Try static file serving first
                std::string static_content_type;
                std::string static_body;
                uint64_t static_length = 0;
                
                if (method == "GET" && try_serve_static(path, static_content_type, static_body, static_length)) {
                    // Serve static file
                    char header_buf[1024];
                    int header_len = snprintf(header_buf, sizeof(header_buf),
                        "HTTP/1.1 200 OK\r\n"
                        "Content-Type: %s\r\n"
                        "Content-Length: %llu\r\n"
                        "Cache-Control: max-age=3600\r\n"
                        "Access-Control-Allow-Origin: *\r\n"
                        "Connection: close\r\n\r\n",
                        static_content_type.c_str(),
                        static_length);
                    send(client, header_buf, header_len, 0);
                    send(client, static_body.c_str(), static_cast<int>(static_length), 0);
                    closesocket(client);
                    return;
                }
                
                // Handle API request
                std::string response_body;
                std::function<void(const std::string&)> stream_callback;
                
                if (is_streaming) {
                    // Set up SSE streaming — send headers BEFORE any data
                    std::string response_headers = 
                        "HTTP/1.1 200 OK\r\n"
                        "Content-Type: text/event-stream\r\n"
                        "Cache-Control: no-cache\r\n"
                        "Connection: keep-alive\r\n"
                        "Access-Control-Allow-Origin: *\r\n\r\n";
                    send(client, response_headers.c_str(), 
                         static_cast<int>(response_headers.size()), 0);
                    sse_headers_sent = true;
                    
                    stream_callback = [client](const std::string& chunk) {
                        std::string sse = "data: " + chunk + "\n\n";
                        send(client, sse.c_str(), static_cast<int>(sse.size()), 0);
                    };
                }
                
                handle_request(method, path, body, response_body, is_streaming, stream_callback);
                
                if (!is_streaming) {
                    // Determine content type
                    const char* ct = "application/json";
                    if (path == "/" && !static_body.empty()) {
                        ct = static_content_type.c_str();
                    }
                    
                    // Send HTTP response
                    char header_buf[512];
                    int header_len = snprintf(header_buf, sizeof(header_buf),
                        "HTTP/1.1 200 OK\r\n"
                        "Content-Type: %s\r\n"
                        "Content-Length: %zu\r\n"
                        "Access-Control-Allow-Origin: *\r\n"
                        "Connection: close\r\n\r\n",
                        ct, response_body.size());
                    send(client, header_buf, header_len, 0);
                    send(client, response_body.c_str(), 
                         static_cast<int>(response_body.size()), 0);
                } else {
                    // If response_body contains data (e.g., error JSON), send it as SSE
                    if (!response_body.empty() && response_body != "null") {
                        std::string sse_data = "data: " + response_body + "\n\n";
                        send(client, sse_data.c_str(), static_cast<int>(sse_data.size()), 0);
                    }
                    // Send SSE done event
                    std::string done = "data: [DONE]\n\n";
                    send(client, done.c_str(), static_cast<int>(done.size()), 0);
                }
            }
            
            closesocket(client);
        }).detach();
    }
    
    closesocket(server_socket_);
    server_socket_ = INVALID_SOCKET;
    WSACleanup();
}

void HTTPServer::handle_request(const std::string& method, const std::string& path,
                                 const std::string& body, std::string& response,
                                 bool& is_streaming,
                                 std::function<void(const std::string&)> stream_callback) {
    // Build route context
    RouteContext ctx;
    ctx.registry = registry_;
    ctx.downloader = downloader_;
    ctx.engine = inference_;
    
    // Route to appropriate handler
    if (path == "/api/generate" || path == "/api/generate/") {
        response = handle_generate(body, stream_callback);
        is_streaming = true;
    } else if (path == "/api/chat" || path == "/api/chat/") {
        response = handle_chat(body, stream_callback);
        is_streaming = true;
    } else if (path == "/api/tags" || path == "/api/tags/") {
        response = handle_tags(body);
        is_streaming = false;
    } else if (path == "/api/pull" || path == "/api/pull/") {
        response = handle_pull(body, stream_callback);
        is_streaming = true;
    } else if (path == "/api/delete" || path == "/api/delete/") {
        response = handle_delete(body);
        is_streaming = false;
    } else if (path == "/" || path == "/api/version") {
        json j;
        j["version"] = "0.1.0";
        j["name"] = "SuperLlama";
        response = j.dump();
    } else {
        json err;
        err["error"] = std::string("Unknown route: ") + path;
        response = err.dump();
    }
}

std::string HTTPServer::handle_generate(const std::string& body,
                                         std::function<void(const std::string&)> stream) {
    try {
        auto req_json = json::parse(body.empty() ? "{}" : body);
        
        InferenceRequest req;
        req.model_name = req_json.value("model", "");
        req.prompt = req_json.value("prompt", "");
        req.chat_mode = false;
        req.gen_config.stream = req_json.value("stream", true);
        req.gen_config.max_tokens = req_json.value("max_tokens", 256);
        req.gen_config.temperature = req_json.value("temperature", 0.7f);
        req.gen_config.top_p = req_json.value("top_p", 0.9f);
        req.gen_config.top_k = req_json.value("top_k", 40);
        
        if (req_json.contains("system")) {
            req.system_prompt = req_json["system"].get<std::string>();
        }
        
        if (!inference_) {
            json err;
            err["error"] = "Inference engine not initialized";
            return err.dump();
        }
        
        std::string accumulated;
        auto tokens = inference_->generate(req, [&](const InferenceToken& token) {
            if (stream && !token.text.empty()) {
                accumulated += token.text;
                std::string chunk = build_generate_response(token, false);
                stream(chunk);
            }
        });
        
        // Build final response
        json response;
        response["model"] = req.model_name;
        response["response"] = accumulated;
        response["done"] = true;
        return response.dump();
        
    } catch (const std::exception& e) {
        json err;
        err["error"] = std::string("Generate error: ") + e.what();
        return err.dump();
    }
}

std::string HTTPServer::handle_chat(const std::string& body,
                                     std::function<void(const std::string&)> stream) {
    try {
        auto req_json = json::parse(body.empty() ? "{}" : body);
        
        InferenceRequest req;
        req.model_name = req_json.value("model", "");
        req.chat_mode = true;
        req.gen_config.stream = req_json.value("stream", true);
        req.gen_config.max_tokens = req_json.value("max_tokens", 256);
        req.gen_config.temperature = req_json.value("temperature", 0.7f);
        
        for (const auto& msg : req_json["messages"]) {
            InferenceRequest::Message m;
            m.role = msg.value("role", "user");
            m.content = msg.value("content", "");
            req.messages.push_back(m);
        }
        
        if (!inference_) {
            json err;
            err["error"] = "Inference engine not initialized";
            return err.dump();
        }
        
        std::string accumulated;
        auto tokens = inference_->generate(req, [&](const InferenceToken& token) {
            if (stream && !token.text.empty()) {
                accumulated += token.text;
                std::string chunk = build_chat_response(token, false);
                stream(chunk);
            }
        });
        
        json response;
        response["model"] = req.model_name;
        
        json message;
        message["role"] = "assistant";
        message["content"] = accumulated;
        
        response["message"] = message;
        response["done"] = true;
        return response.dump();
        
    } catch (const std::exception& e) {
        json err;
        err["error"] = std::string("Chat error: ") + e.what();
        return err.dump();
    }
}

std::string HTTPServer::handle_tags(const std::string& body) {
    try {
        json response;
        response["models"] = json::array();
        
        if (registry_) {
            auto models = registry_->list_models();
            for (const auto& m : models) {
                json model_json;
                model_json["name"] = m.name;
                model_json["size"] = m.size_bytes;
                model_json["modified_at"] = 
                    std::chrono::duration_cast<std::chrono::seconds>(
                        m.modified_at.time_since_epoch()).count();
                model_json["digest"] = m.digest;
                model_json["format"] = dtype_name(m.quantization);
                model_json["family"] = m.family;
                char param_buf[32];
                snprintf(param_buf, sizeof(param_buf), "%.1fB", m.param_count / 1e9);
                model_json["parameter_size"] = param_buf;
                response["models"].push_back(model_json);
            }
        }
        
        return response.dump();
        
    } catch (const std::exception& e) {
        json err;
        err["error"] = std::string("Tags error: ") + e.what();
        return err.dump();
    }
}

std::string HTTPServer::handle_pull(const std::string& body,
                                     std::function<void(const std::string&)> stream) {
    try {
        auto req_json = json::parse(body.empty() ? "{}" : body);
        std::string model_repo = req_json.value("model", "");
        std::string filename    = req_json.value("file", "");
        
        // Backward compat: if "file" is empty, try extracting from "model" as "repo/filename"
        if (filename.empty() && model_repo.find('/') != std::string::npos) {
            auto last_slash = model_repo.rfind('/');
            filename = model_repo.substr(last_slash + 1);
            model_repo = model_repo.substr(0, last_slash);
        }
        
        if (model_repo.empty()) {
            json err;
            err["error"] = "model repository is required";
            return err.dump();
        }
        if (filename.empty()) {
            json err;
            err["error"] = "GGUF filename is required";
            return err.dump();
        }
        
        // Build HuggingFace download URL: https://huggingface.co/{org}/{repo}/resolve/main/{file}
        std::string download_url = "https://huggingface.co/" + model_repo + "/resolve/main/" + filename;
        
        // Build a safe local name: replace / and : with -
        std::string safe_name = model_repo + "-" + filename;
        for (auto& c : safe_name) {
            if (c == '/' || c == ':') c = '-';
        }
        // If safe_name already ends with .gguf don't double it
        if (safe_name.size() < 5 || safe_name.substr(safe_name.size() - 5) != ".gguf") {
            safe_name += ".gguf";
        }
        
        // Ensure models directory exists
        std::string models_dir = "./models";
        std::filesystem::path dest_dir(models_dir);
        if (!std::filesystem::exists(dest_dir)) {
            std::filesystem::create_directories(dest_dir);
        }
        std::string dest_path = (dest_dir / safe_name).string();
        
        SL_LOG_INFO("Downloading model from {} to {}", download_url, dest_path);
        
        if (stream) {
            json progress;
            progress["status"] = "preparing";
            progress["model"] = safe_name;
            stream(progress.dump());
        }
        
        if (downloader_) {
            bool started = downloader_->download(
                safe_name, download_url, dest_path,
                [stream](const DownloadProgress& p) {
                    if (!stream) return;
                    json status;
                    status["status"] = p.status;
                    status["total"] = p.total_bytes;
                    status["completed"] = p.downloaded_bytes;
                    if (!p.error_message.empty()) {
                        status["error"] = p.error_message;
                    }
                    if (p.completed) {
                        status["status"] = "success";
                    }
                    stream(status.dump());
                }
            );
            
            if (started) {
                // Download runs in background — keep the connection alive and stream final result
                // Poll for completion (max 10 min)
                auto start = std::chrono::steady_clock::now();
                while (std::chrono::steady_clock::now() - start < std::chrono::minutes(10)) {
                    auto opt_progress = downloader_->get_progress(safe_name);
                    if (opt_progress) {
                        const auto& p = *opt_progress;
                        if (p.completed) {
                            if (stream) {
                                json done;
                                done["status"] = "success";
                                done["model"] = safe_name;
                                stream(done.dump());
                            }
                            json response;
                            response["status"] = "success";
                            response["model"] = safe_name;
                            return response.dump();
                        }
                        if (p.status == "error" || p.status == "cancelled") {
                            json err;
                            err["error"] = p.error_message.empty() ? p.status : p.error_message;
                            return err.dump();
                        }
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(250));
                }
                
                // Timed out
                json response;
                response["status"] = "downloading";
                response["model"] = safe_name;
                response["message"] = "Download still in progress (large file)";
                return response.dump();
            } else {
                json err;
                err["error"] = "Failed to start download (may already be in progress)";
                return err.dump();
            }
        }
        
        json err;
        err["error"] = "Downloader not available";
        return err.dump();
        
    } catch (const std::exception& e) {
        json err;
        err["error"] = std::string("Pull error: ") + e.what();
        return err.dump();
    }
}

std::string HTTPServer::handle_delete(const std::string& body) {
    try {
        auto req_json = json::parse(body.empty() ? "{}" : body);
        std::string model_name = req_json.value("model", "");
        
        if (registry_) {
            bool deleted = registry_->delete_model(model_name);
            json response;
            response["deleted"] = deleted;
            return response.dump();
        }
        
        json err;
        err["error"] = "Model registry not initialized";
        return err.dump();
        
    } catch (const std::exception& e) {
        json err;
        err["error"] = std::string("Delete error: ") + e.what();
        return err.dump();
    }
}

std::string HTTPServer::build_generate_response(const InferenceToken& token, bool include_context) {
    json j;
    j["model"] = "model";
    j["response"] = token.text;
    j["done"] = token.is_final;
    return j.dump();
}

std::string HTTPServer::build_chat_response(const InferenceToken& token, bool include_context) {
    json j;
    j["model"] = "model";
    
    json msg;
    msg["role"] = "assistant";
    msg["content"] = token.text;
    
    j["message"] = msg;
    j["done"] = token.is_final;
    return j.dump();
}

} // namespace sl