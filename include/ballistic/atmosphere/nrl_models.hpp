#pragma once

// =============================================================================
// NRL models — NRLMSIS 2.1 and HWM14, loaded at run time from a plugin
// =============================================================================
//
// Model-based drag (make_drag_environment, the from_models_* tables) needs the
// NRLMSIS 2.1 atmosphere and the HWM14 wind model. They are not compiled into
// the library: NRLMSIS 2.1 is licensed for research, academic and non-profit
// use only (NRL-SOF-014-1), so they ship as a separate shared library,
// ballistic_nrl, that a user installs knowingly:
//
//   Python  ballistic.install_nrl_models(accept_license=True) downloads it into
//           the data directory (or installs a file already downloaded).
//   C++     build the ballistic_nrl target (same source tree) and point the
//           library at it: InitConfig::nrl_models_path, the environment
//           variable BALLISTIC_NRL_MODELS, or copy it to default_install_path().
//
// Without it, everything except model-based drag works; drag from measured
// density (from_accelerometer_density) needs no NRL code.
//
// Lookup order on first use (require() / try_autoload()): a path already
// loaded; BALLISTIC_NRL_MODELS; default_install_path(). A loaded plugin stays
// loaded for the life of the process.
// =============================================================================

#include "nrl_plugin_abi.h"
#include <filesystem>
#include <string>
#include <string_view>

namespace ballistic {
namespace atmosphere {
namespace nrl {

/// What is loaded, and from where.
struct Status {
    bool loaded = false;
    std::string path;     // the plugin file; empty when nothing is loaded
    std::string models;   // e.g. "NRLMSIS 2.1, HWM14"; empty when nothing is loaded
    std::string problem;  // why the last load attempt failed; empty if none failed
};

/// File name of the plugin on this platform: ballistic_nrl.so (Linux),
/// ballistic_nrl.dylib (macOS), ballistic_nrl.dll (Windows).
[[nodiscard]] std::string_view library_file_name() noexcept;

/// Where install_nrl_models puts the plugin and where it is looked for:
/// <data directory>/nrl/<library_file_name()>.
[[nodiscard]] std::filesystem::path default_install_path();

/// Load the plugin from `path`. Throws ConfigError when the file is missing,
/// is not a loadable library, lacks the plugin's functions or was built for
/// another interface version. Loading a second path once one is loaded
/// throws ConfigError too (a process uses one copy of the models).
void load(const std::filesystem::path& path);

/// Load from BALLISTIC_NRL_MODELS or default_install_path() if nothing is
/// loaded yet. Returns whether the models are loaded afterwards; a file that
/// is present but cannot be loaded is recorded in status().problem.
bool try_autoload();

/// The current state (does not try to load anything).
[[nodiscard]] Status status();

/// Make sure the models are loaded (try_autoload), else throw ConfigError
/// naming `caller` and saying how to install them.
void require(std::string_view caller);

/// NRLMSIS 2.1 at one point. require() must have succeeded.
[[nodiscard]] BallisticNrlMsisOutput msis(const BallisticNrlMsisInput& in);

/// HWM14 wind at one point: {meridional (north +), zonal (east +)} [m/s].
/// require() must have succeeded.
void hwm14(const BallisticNrlHwmInput& in, float wind[2]);

}  // namespace nrl
}  // namespace atmosphere
}  // namespace ballistic
