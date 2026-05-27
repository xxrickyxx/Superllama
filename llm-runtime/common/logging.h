#pragma once
#include "common/platform.h"
#include "types.h"
#include <string>
#include <sstream>
#include <iostream>
#include <chrono>
#include <mutex>
#include <format>
#include <filesystem>
#include <ctime>

namespace sl {

class Logger {
public:
    static Logger& instance() {
        static Logger logger;
        return logger;
    }
    
    void set_level(LogLevel level) { min_level_ = level; }
    LogLevel level() const { return min_level_; }
    
    template<typename... Args>
    void log(LogLevel lvl, const std::string& file, int line,
             const std::string& fmt, Args&&... args) {
        if (lvl < min_level_) return;
        
        std::lock_guard lock(mutex_);
        
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        
        std::string level_str;
        switch (lvl) {
            case LogLevel::TRACE: level_str = "TRACE"; break;
            case LogLevel::DEBUG: level_str = "DEBUG"; break;
            case LogLevel::INFO:  level_str = "INFO";  break;
            case LogLevel::WARN:  level_str = "WARN";  break;
            case LogLevel::ERROR: level_str = "ERROR"; break;
            case LogLevel::FATAL: level_str = "FATAL"; break;
        }
        
        std::string filename = std::filesystem::path(file).filename().string();
        
        char time_buf[32];
        struct tm tm_info;
        localtime_s(&tm_info, &time);
        strftime(time_buf, sizeof(time_buf), "%H:%M:%S", &tm_info);
        
        try {
            auto msg = std::vformat(fmt, std::make_format_args(args...));
            std::cout << std::format("[{}] {} ({}:{}): {}\n", level_str, time_buf, filename, line, msg);
        } catch (...) {
            std::cout << "[" << level_str << "] " << filename << ":" << line << ": <format error>" << std::endl;
        }
    }
    
private:
    LogLevel min_level_ = LogLevel::INFO;
    std::mutex mutex_;
};

} // namespace sl

#define SL_LOG_TRACE(fmt, ...) sl::Logger::instance().log(sl::LogLevel::TRACE, __FILE__, __LINE__, fmt __VA_OPT__(,) __VA_ARGS__)
#define SL_LOG_DEBUG(fmt, ...) sl::Logger::instance().log(sl::LogLevel::DEBUG, __FILE__, __LINE__, fmt __VA_OPT__(,) __VA_ARGS__)
#define SL_LOG_INFO(fmt, ...)  sl::Logger::instance().log(sl::LogLevel::INFO,  __FILE__, __LINE__, fmt __VA_OPT__(,) __VA_ARGS__)
#define SL_LOG_WARN(fmt, ...)  sl::Logger::instance().log(sl::LogLevel::WARN,  __FILE__, __LINE__, fmt __VA_OPT__(,) __VA_ARGS__)
#define SL_LOG_ERROR(fmt, ...) sl::Logger::instance().log(sl::LogLevel::ERROR, __FILE__, __LINE__, fmt __VA_OPT__(,) __VA_ARGS__)
#define SL_LOG_FATAL(fmt, ...) sl::Logger::instance().log(sl::LogLevel::FATAL, __FILE__, __LINE__, fmt __VA_OPT__(,) __VA_ARGS__)