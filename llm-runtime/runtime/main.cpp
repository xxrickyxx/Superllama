#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif
#include "common/types.h"
#include "common/logging.h"
#include "engine/inference.h"
#include "engine/model_loader.h"
#include "engine/paging_manager.h"
#include "models/model_registry.h"
#include "runtime/server.h"

#include <iostream>
#include <string>
#include <csignal>

using namespace sl;

// Global state for signal handling
static std::atomic<bool> g_running{true};
static HTTPServer* g_server = nullptr;

void signal_handler(int signal) {
    SL_LOG_INFO("Received signal {}, shutting down...", signal);
    g_running = false;
    if (g_server) {
        g_server->stop();
    }
}

void print_banner() {
    std::cout << R"(
   _____                      __    __                                
  / ___/__  ______  ___  _____/ /   / /   ____ ___  ____ ___  ____ _  
  \__ \/ / / / __ \/ _ \/ ___/ /   / /   / __ `__ \/ __ `__ \/ __ `/  
 ___/ / /_/ / /_/ /  __/ /  / /___/ /___/ / / / / / / / / / / /_/ /   
/____/\__,_/ .___/\___/_/  /_____/_____/_/ /_/ /_/_/ /_/ /_/\__,_/    
          /_/                                                          

    High-Performance LLM Inference Runtime
    SSD-Paged Execution | GGUF Compatible | Ollama API
    Version 0.1.0
    )" << std::endl;
}

void print_usage() {
    std::cout << R"(
Usage: superllama.exe [OPTIONS]

Options:
  --model <path>     Path to GGUF model file
  --port <port>      API server port (default: 11434)
  --host <host>      Bind address (default: 127.0.0.1)
  --cache <mb>       RAM cache size in MB (default: 4096)
  --threads <n>      Number of CPU threads (default: auto)
  --models-path <dir> Directory to scan for models
  --list-models      List installed models and exit
  --no-server        Don't start API server
  --verbose          Enable debug logging
  --help             Show this help

Examples:
  superllama.exe --model C:\models\llama3.gguf
  superllama.exe --port 8080 --cache 8192
  superllama.exe --list-models
  superllama.exe
    )" << std::endl;
}

int main(int argc, char* argv[]) {
    // Parse command line
    std::string model_path;
    uint16_t port = 11434;
    std::string host = "127.0.0.1";
    size_t cache_mb = 4096;
    uint32_t num_threads = 0;
    std::string models_path = "./models";
    bool list_models_only = false;
    bool no_server = false;
    bool verbose = false;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            port = static_cast<uint16_t>(std::stoul(argv[++i]));
        } else if (arg == "--host" && i + 1 < argc) {
            host = argv[++i];
        } else if (arg == "--cache" && i + 1 < argc) {
            cache_mb = std::stoul(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            num_threads = std::stoul(argv[++i]);
        } else if (arg == "--models-path" && i + 1 < argc) {
            models_path = argv[++i];
        } else if (arg == "--list-models") {
            list_models_only = true;
        } else if (arg == "--no-server") {
            no_server = true;
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--help") {
            print_usage();
            return 0;
        }
    }
    
    if (verbose) {
        Logger::instance().set_level(LogLevel::DEBUG);
    }
    
    print_banner();
    
    // Setup signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    
    // Configure server
    ServerConfig server_config;
    server_config.port = port;
    server_config.host = host;
    server_config.num_threads = num_threads;
    server_config.model_paths = {models_path, "./models", 
                                  std::string(std::getenv("USERPROFILE") ? std::getenv("USERPROFILE") : "C:/Users/Default") + "/.cache/superllama/models"};
    
    // Initialize model registry
    ModelRegistry registry(server_config);
    registry.scan();
    
    // Initialize model downloader
    ModelDownloader downloader;
    
    // List models if requested
    if (list_models_only) {
        auto models = registry.list_models();
        if (models.empty()) {
            std::cout << "\nNo models found in configured directories.\n";
            std::cout << "Place .gguf files in: " << models_path << "\n";
            std::cout << "Or download models with: superllama pull <model_name>\n";
        } else {
            std::cout << "\nInstalled Models:\n";
            std::cout << std::string(80, '-') << "\n";
            for (const auto& m : models) {
                char buf[128];
                snprintf(buf, sizeof(buf), "  %-30s %-10s %-8s %s\n",
                    m.name.c_str(), m.size_str().c_str(),
                    arch_name(m.architecture), dtype_name(m.quantization));
                std::cout << buf;
            }
            std::cout << std::string(80, '-') << "\n";
            std::cout << "  Total: " << models.size() << " models\n";
        }
        return 0;
    }
    
    // Initialize paging system
    SSDPagingManager paging_mgr;
    
    // Initialize inference engine
    InferenceEngine inference;
    ModelLoader model_loader(paging_mgr);
    
    // Load model if specified
    if (!model_path.empty()) {
        SL_LOG_INFO("Loading model: {}", model_path);
        
        if (!std::filesystem::exists(model_path)) {
            SL_LOG_FATAL("Model file not found: {}", model_path);
            return 1;
        }
        
        auto config = model_loader.load(model_path);
        if (!inference.initialize(model_loader)) {
            SL_LOG_FATAL("Failed to initialize inference engine");
            return 1;
        }
        
        // Log memory configuration
        auto stats = paging_mgr.get_stats();
        bool fits_in_ram = paging_mgr.model_fits_in_ram();
        
        SL_LOG_INFO("Model: {} | {} layers | {:.1f} GB",
                    config.name, config.n_layer,
                    config.file_size_bytes / 1e9);
        SL_LOG_INFO("RAM cache: {} MB | SSD streaming: {}",
                    cache_mb,
                    fits_in_ram ? "not needed (model fits in RAM)" : "enabled");
        
        // Warmup
        SL_LOG_INFO("Warming up inference engine...");
        inference.warmup();
        SL_LOG_INFO("Warmup complete");
    } else {
        SL_LOG_INFO("No model specified. Server will start in model-management mode.");
        SL_LOG_INFO("Use the API to load/select models.");
    }
    
    // Start API server
    if (!no_server) {
        HTTPServer server(server_config);
        server.set_inference_engine(&inference);
        server.set_model_registry(&registry);
        server.set_model_downloader(&downloader);
        
        // Serve the UI from the same origin to avoid CORS/file:// issues
        std::string ui_path = "../ui";
        if (!std::filesystem::exists(ui_path)) {
            ui_path = "../../ui";  // fallback when run from build/
        }
        server.set_ui_directory(ui_path);
        
        g_server = &server;
        
        if (!server.start()) {
            SL_LOG_FATAL("Failed to start HTTP server");
            return 1;
        }
        
        std::cout << "\n  API Server: http://" << host << ":" << port << std::endl;
        std::cout << "  Endpoints:" << std::endl;
        std::cout << "    POST /api/generate  - Generate text" << std::endl;
        std::cout << "    POST /api/chat      - Chat completions" << std::endl;
        std::cout << "    GET  /api/tags      - List models" << std::endl;
        std::cout << "    POST /api/pull      - Download model" << std::endl;
        std::cout << "    DELETE /api/delete  - Remove model" << std::endl;
        std::cout << "\n  Press Ctrl+C to stop\n" << std::endl;
        
        // Main loop - wait for shutdown signal
        while (g_running && server.is_running()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        server.stop();
        g_server = nullptr;
    }
    
    SL_LOG_INFO("SuperLlama shutdown complete");
    return 0;
}
