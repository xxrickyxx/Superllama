#pragma once

#ifdef _WIN32
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <windows.h>
  // These macros STILL get defined by some MSVC headers despite NOMINMAX.
  // Undefine them unconditionally to be safe.
  #ifdef max
    #undef max
  #endif
  #ifdef min
    #undef min
  #endif
  #ifdef FATAL
    #undef FATAL
  #endif
  #ifdef ERROR
    #undef ERROR
  #endif
  #ifdef TRACE
    #undef TRACE
  #endif
#endif
