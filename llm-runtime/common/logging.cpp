#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif
#include "logging.h"

// Logger is fully inline/header-based.
// This file exists to satisfy CMake dependency tracking.
namespace sl {

} // namespace sl