#pragma once
#include "common/platform.h"

#include "common/types.h"
#include <string>
#include <functional>
#include <nlohmann/json.hpp>

namespace sl {

using json = nlohmann::json;

// ====================================================================
// API Route Handler definitions
// Each route returns JSON and optionally streams tokens
// ====================================================================

class InferenceEngine;  // forward decl
class ModelRegistry;     // forward decl
class ModelDownloader;   // forward decl

struct RouteContext {
    InferenceEngine* engine = nullptr;
    ModelRegistry* registry = nullptr;
    ModelDownloader* downloader = nullptr;
};

using StreamCallback = std::function<void(const std::string&)>;
using RouteHandler = std::function<std::string(const json& request, RouteContext ctx, StreamCallback stream)>;

// Route registration
class APIRouter {
public:
    APIRouter();
    
    void register_route(const std::string& method, const std::string& path, RouteHandler handler);
    
    std::string dispatch(const std::string& method, const std::string& path, 
                         const std::string& body, RouteContext ctx,
                         bool is_streaming, StreamCallback stream);
    
private:
    struct Route {
        std::string method;
        std::string path;
        RouteHandler handler;
    };
    std::vector<Route> routes_;
};

// Pre-built handler implementations
namespace api_handlers {

std::string generate(const json& req, RouteContext ctx, StreamCallback stream);
std::string chat(const json& req, RouteContext ctx, StreamCallback stream);
std::string tags(const json& req, RouteContext ctx, StreamCallback stream);
std::string pull(const json& req, RouteContext ctx, StreamCallback stream);
std::string delete_model(const json& req, RouteContext ctx, StreamCallback stream);
std::string show(const json& req, RouteContext ctx, StreamCallback stream);
std::string embeddings(const json& req, RouteContext ctx, StreamCallback stream);
std::string version(const json& req, RouteContext ctx, StreamCallback stream);

} // namespace api_handlers

} // namespace sl