#pragma once

#include "../RocketSim/src/RocketSim.h"
#include "../RocketSim/src/Sim/GameEventTracker/GameEventTracker.h"

// Use RocketSim namespace
using namespace RocketSim;

#include "AsyncLogger.h"
#include <sstream>
#include <thread>

// Define our own log
#define RG_LOG(s)                                                              \
  {                                                                            \
    std::ostringstream _oss;                                                   \
    _oss << s;                                                                 \
    RLGC::AsyncLogger::LogDirect(_oss.str());                                  \
  }

#define RG_NO_COPY(className)                                                  \
  className(const className &) = delete;                                       \
  className &operator=(const className &) = delete

#define RG_ERR_CLOSE(s)                                                        \
  {                                                                            \
    std::ostringstream _oss;                                                   \
    _oss << "RG FATAL ERROR: " << s << "\n"                                    \
         << "  at " << __FILE__ << ":" << __LINE__ << " ("                   \
         << __func__ << ")\n"                                                  \
         << "  thread " << std::this_thread::get_id();                         \
    std::string _errorStr = _oss.str();                                        \
    RG_LOG(_errorStr);                                                         \
    throw std::runtime_error(_errorStr);                                       \
    exit(EXIT_FAILURE);                                                        \
  }

#ifndef RG_UNSAFE
#define RG_ASSERT(cond)                                                        \
  {                                                                            \
    if (!(cond)) {                                                             \
      RG_ERR_CLOSE("Assertion failed: " << #cond);                             \
    }                                                                          \
  }
#else
#define RG_ASSERT(cond)                                                        \
  {                                                                            \
  }
#endif

#define RG_DIVIDER std::string(40, '=')
