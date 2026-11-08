// =============================================================================
// NRL models plugin loader — implementation (see nrl_models.hpp)
// =============================================================================

#include "ballistic/atmosphere/nrl_models.hpp"
#include "ballistic/core/data_paths.hpp"
#include "ballistic/core/errors.hpp"

#include <atomic>
#include <cstdlib>
#include <mutex>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace ballistic {
namespace atmosphere {
namespace nrl {

namespace {

struct Api {
    int (*abi_version)(void) = nullptr;
    const char* (*models)(void) = nullptr;
    int (*msis)(const BallisticNrlMsisInput*, BallisticNrlMsisOutput*) = nullptr;
    int (*hwm14)(const BallisticNrlHwmInput*, float*) = nullptr;
    const char* (*last_error)(void) = nullptr;
};

// Written once under g_mutex, then read without a lock (acquire/release).
Api g_api;
std::atomic<bool> g_loaded{false};
std::mutex g_mutex;
std::string g_path;
std::string g_problem;

constexpr const char* kReleasesUrl = "https://github.com/nvhuynh16/ballistic/releases";

#if defined(_WIN32)
using Handle = HMODULE;
Handle open_library(const std::filesystem::path& p, std::string& err) {
    Handle h = LoadLibraryW(p.wstring().c_str());
    if (!h) err = "LoadLibrary failed with error " + std::to_string(GetLastError());
    return h;
}
void* symbol(Handle h, const char* name) {
    return reinterpret_cast<void*>(GetProcAddress(h, name));
}
#else
using Handle = void*;
Handle open_library(const std::filesystem::path& p, std::string& err) {
    Handle h = dlopen(p.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        const char* e = dlerror();
        err = e ? e : "dlopen failed";
    }
    return h;
}
void* symbol(Handle h, const char* name) { return dlsym(h, name); }
#endif

template <class F>
bool bind(Handle h, const char* name, F& fn) {
    fn = reinterpret_cast<F>(symbol(h, name));
    return fn != nullptr;
}

// Load `path` into g_api. Caller holds g_mutex and has checked !g_loaded.
// Returns an empty string on success, else the reason.
std::string load_locked(const std::filesystem::path& path) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec)) {
        return "no file at " + path.string();
    }
    std::string err;
    Handle h = open_library(path, err);
    if (!h) return path.string() + " is not a loadable library: " + err;
    Api api;
    if (!bind(h, "ballistic_nrl_abi_version", api.abi_version) ||
        !bind(h, "ballistic_nrl_models", api.models) ||
        !bind(h, "ballistic_nrl_msis", api.msis) ||
        !bind(h, "ballistic_nrl_hwm14", api.hwm14) ||
        !bind(h, "ballistic_nrl_last_error", api.last_error)) {
        return path.string() + " is not the ballistic NRL models plugin (its functions are missing)";
    }
    if (const int v = api.abi_version(); v != BALLISTIC_NRL_ABI_VERSION) {
        return path.string() + " was built for plugin interface version " + std::to_string(v) +
               ", this library needs version " + std::to_string(BALLISTIC_NRL_ABI_VERSION) +
               ": install the plugin that matches this version of the library";
    }
    // The library handle is never closed: the models stay loaded for the life
    // of the process, and function pointers into it may be held anywhere.
    g_api = api;
    g_path = path.string();
    g_problem.clear();
    g_loaded.store(true, std::memory_order_release);
    return {};
}

std::filesystem::path env_path() {
    const char* p = std::getenv("BALLISTIC_NRL_MODELS");
    return (p && *p) ? std::filesystem::path(p) : std::filesystem::path{};
}

}  // namespace

std::string_view library_file_name() noexcept {
#if defined(_WIN32)
    return "ballistic_nrl.dll";
#elif defined(__APPLE__)
    return "ballistic_nrl.dylib";
#else
    return "ballistic_nrl.so";
#endif
}

std::filesystem::path default_install_path() {
    return get_default_data_directory() / "nrl" / std::string(library_file_name());
}

void load(const std::filesystem::path& path) {
    std::lock_guard lock(g_mutex);
    if (g_loaded.load(std::memory_order_acquire)) {
        std::error_code ec;
        if (std::filesystem::equivalent(path, g_path, ec)) return;
        throw ConfigError("the NRL models are already loaded from " + g_path +
                          "; a process uses one copy, so " + path.string() +
                          " was not loaded (restart the process to switch)");
    }
    if (std::string why = load_locked(path); !why.empty()) {
        g_problem = why;
        throw ConfigError("cannot load the NRL models plugin: " + why);
    }
}

bool try_autoload() {
    if (g_loaded.load(std::memory_order_acquire)) return true;
    std::lock_guard lock(g_mutex);
    if (g_loaded.load(std::memory_order_acquire)) return true;
    // An explicitly named file that cannot be loaded is a problem to report;
    // an absent default install is simply "not installed".
    if (const auto p = env_path(); !p.empty()) {
        if (std::string why = load_locked(p); !why.empty()) {
            g_problem = "BALLISTIC_NRL_MODELS: " + why;
            return false;
        }
        return true;
    }
    const auto p = default_install_path();
    std::error_code ec;
    if (!std::filesystem::exists(p, ec)) return false;
    if (std::string why = load_locked(p); !why.empty()) {
        g_problem = why;
        return false;
    }
    return true;
}

Status status() {
    std::lock_guard lock(g_mutex);
    Status s;
    s.loaded = g_loaded.load(std::memory_order_acquire);
    s.path = g_path;
    s.problem = g_problem;
    if (s.loaded) s.models = g_api.models();
    return s;
}

void require(std::string_view caller) {
    if (try_autoload()) return;
    std::string problem;
    {
        std::lock_guard lock(g_mutex);
        problem = g_problem;
    }
    std::string msg =
        std::string(caller) +
        ": model-based drag needs the NRLMSIS 2.1 atmosphere and HWM14 wind models, which are "
        "not installed. They are distributed separately because NRLMSIS 2.1 is licensed by the "
        "U.S. Naval Research Laboratory for research, academic and non-profit use only "
        "(NRL-SOF-014-1; commercial use is not permitted). To install them: Python, "
        "ballistic.install_nrl_models(accept_license=True) downloads them (offline, download "
        "the plugin for your platform from " + std::string(kReleasesUrl) +
        " and pass it as source=); or copy the plugin to " + default_install_path().string() +
        " or set BALLISTIC_NRL_MODELS to its path; C++, build the ballistic_nrl target and set "
        "InitConfig::nrl_models_path. Without them, propagate without drag or build drag from "
        "measured density (from_accelerometer_density).";
    if (!problem.empty()) msg += " A plugin was found but not loaded: " + problem + ".";
    throw ConfigError(msg);
}

BallisticNrlMsisOutput msis(const BallisticNrlMsisInput& in) {
    if (!g_loaded.load(std::memory_order_acquire)) require("NRLMSIS 2.1");
    BallisticNrlMsisOutput out{};
    if (g_api.msis(&in, &out) != 0) {
        throw ConfigError(std::string("NRLMSIS 2.1 (plugin) failed: ") + g_api.last_error());
    }
    return out;
}

void hwm14(const BallisticNrlHwmInput& in, float wind[2]) {
    if (!g_loaded.load(std::memory_order_acquire)) require("HWM14");
    if (g_api.hwm14(&in, wind) != 0) {
        throw ConfigError(std::string("HWM14 (plugin) failed: ") + g_api.last_error());
    }
}

}  // namespace nrl
}  // namespace atmosphere
}  // namespace ballistic
