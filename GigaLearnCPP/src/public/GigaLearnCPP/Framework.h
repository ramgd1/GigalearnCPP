#pragma once

// Include RLGymCPP
#include <RLGymCPP/BasicTypes/Lists.h>
#include <RLGymCPP/EnvSet/EnvSet.h>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <thread>
#include <vector>


using RLGC::FList;
using RLGC::IList;

#if defined(_MSC_VER)
// MSVC
#define RG_EXPORTED __declspec(dllexport)
#define RG_IMPORTED __declspec(dllimport)
#pragma warning(disable : 4251)
#else
// Everything else (?)
#define RG_EXPORTED __attribute__((visibility("default")))
#define RG_IMPORTED
#endif

#ifdef WITHIN_GGL
#define RG_IMEXPORT RG_EXPORTED
#else
#define RG_IMEXPORT RG_IMPORTED
#endif

#define RG_SLEEP(ms) std::this_thread::sleep_for(std::chrono::milliseconds(ms))

#define THREAD_WAIT() RG_SLEEP(2)
