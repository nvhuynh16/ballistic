// Linked into every test and diagnostic executable (tests/CMakeLists.txt):
// points the library at the NRL models plugin built in this tree, unless
// BALLISTIC_NRL_MODELS already names one, so model-based drag works whether a
// binary runs under ctest or by hand. It only sets an environment variable
// during static initialization; the library reads it on first use of drag.

#include <cstdlib>

namespace {

struct UseBuiltNrlPlugin {
    UseBuiltNrlPlugin() {
        const char* p = std::getenv("BALLISTIC_NRL_MODELS");
        if (p != nullptr && *p != '\0') return;
#ifdef _WIN32
        _putenv_s("BALLISTIC_NRL_MODELS", BALLISTIC_TEST_NRL_PLUGIN);
#else
        setenv("BALLISTIC_NRL_MODELS", BALLISTIC_TEST_NRL_PLUGIN, 1);
#endif
    }
};

const UseBuiltNrlPlugin use_built_nrl_plugin;

}  // namespace
