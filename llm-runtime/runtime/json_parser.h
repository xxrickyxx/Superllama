#pragma once
#include "common/platform.h"

#include <string>
#include <nlohmann/json.hpp>

namespace sl {

// Thin wrapper for nlohmann::json to avoid exposing it everywhere
using Json = nlohmann::json;

inline Json parse_json(const std::string& str) {
    return Json::parse(str);
}

inline std::string dump_json(const Json& j, int indent = -1) {
    return j.dump(indent);
}

} // namespace sl
