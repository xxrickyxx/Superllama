cmake_minimum_required(VERSION 3.20)
project(SuperLlama VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Windows-specific optimizations
if(WIN32)
    add_definitions(-DWIN32_LEAN_AND_MEAN -DNOMINMAX -D_WIN32_WINNT=0x0A00)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# SIMD detection
include(CheckCXXCompilerFlag)
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2 /fp:fast /O2 /GL /Zc:preprocessor")
    add_link_options(/LTCG)
else()
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    if(COMPILER_SUPPORTS_AVX2)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -O3 -march=native")
    endif()
endif()

# Dependencies
# Add vcpkg paths for proper package finding
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../vcpkg/installed/x64-windows")

find_package(CURL CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP)

# Source files
set(COMMON_SOURCES
    common/logging.cpp
)

set(CORE_SOURCES
    core/tensor.cpp
    core/attention.cpp
    core/mlp.cpp
    core/transformer_block.cpp
)

set(ENGINE_SOURCES
    engine/model_loader.cpp
    engine/paging_manager.cpp
    engine/kv_cache.cpp
    engine/inference.cpp
    engine/ssd_streamer.cpp
)

set(RUNTIME_SOURCES
    runtime/server.cpp
    runtime/api_routes.cpp
    runtime/json_parser.cpp
)

set(MODELS_SOURCES
    models/model_registry.cpp
    models/downloader.cpp
    models/model_index.cpp
)

# Core library
add_library(superllama_core STATIC
    ${COMMON_SOURCES}
    ${CORE_SOURCES}
)
target_include_directories(superllama_core PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(superllama_core PUBLIC Threads::Threads nlohmann_json::nlohmann_json)

# Engine library
add_library(superllama_engine STATIC
    ${ENGINE_SOURCES}
)
target_include_directories(superllama_engine PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(superllama_engine PUBLIC superllama_core)

# Models library
add_library(superllama_models STATIC
    ${MODELS_SOURCES}
)
target_include_directories(superllama_models PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(superllama_models PUBLIC superllama_core CURL::libcurl)

# Runtime server
add_executable(superllama_server
    runtime/main.cpp
    ${RUNTIME_SOURCES}
)
target_include_directories(superllama_server PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(superllama_server PRIVATE 
    superllama_engine 
    superllama_models 
    superllama_core
    CURL::libcurl
    nlohmann_json::nlohmann_json
    ws2_32
    winhttp
)

# Tests
enable_testing()
add_executable(superllama_tests
    tests/test_main.cpp
    tests/test_tensor.cpp
    tests/test_paging.cpp
)
target_link_libraries(superllama_tests PRIVATE superllama_core superllama_engine)