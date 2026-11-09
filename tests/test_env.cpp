// Linked into every test and diagnostic executable (tests/CMakeLists.txt) so
// a binary sees the same data whether it runs under ctest or by hand, on any
// machine. During static initialization, unless already set, it points
//   BALLISTIC_DATA_DIR   at tests/data/eop/ (the committed finals2000A.all),
//                        so initialize({}) finds the same Earth orientation
//                        data everywhere and never the user's own data
//                        directory (tests that need an empty one use
//                        helpers::IsolatedDataDir);
//   BALLISTIC_NRL_MODELS at the NRL models plugin built in this tree, when
//                        it is built, so model-based drag works.
// The library reads both variables on first use.

#include <cstdlib>

namespace {

void set_unless_set(const char* name, const char* value) {
    const char* p = std::getenv(name);
    if (p != nullptr && *p != '\0') return;
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

struct TestEnvironment {
    TestEnvironment() {
        set_unless_set("BALLISTIC_DATA_DIR", BALLISTIC_TEST_DATA_DIR);
#ifdef BALLISTIC_TEST_NRL_PLUGIN
        set_unless_set("BALLISTIC_NRL_MODELS", BALLISTIC_TEST_NRL_PLUGIN);
#endif
    }
};

const TestEnvironment test_environment;

}  // namespace
