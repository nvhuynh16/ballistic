// Error hierarchy and the diagnostics handler that turns library diagnostics
// (non-fatal notes such as "data held past the end of a table") into Python
// warnings (core/errors.hpp, core/diagnostics.hpp).

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <ballistic/core/diagnostics.hpp>
#include <ballistic/core/errors.hpp>

#include <string>
#include <unordered_map>

namespace nb = nanobind;
using namespace ballistic;

namespace {

// Category name -> Python warning class, set once by ballistic/__init__.py.
// Leaked on purpose: it must outlive every C++ caller, including ones that
// report during interpreter shutdown (which are then dropped, see below).
std::unordered_map<std::string, nb::object>* g_warning_classes = nullptr;

void python_handler(const diagnostics::Diagnostic& d) {
#if PY_VERSION_HEX >= 0x030D0000
    if (!Py_IsInitialized() || Py_IsFinalizing()) return;
#else
    if (!Py_IsInitialized() || _Py_IsFinalizing()) return;
#endif
    nb::gil_scoped_acquire gil;
    PyObject* cls = PyExc_RuntimeWarning;
    if (g_warning_classes) {
        auto it = g_warning_classes->find(std::string(diagnostics::category_name(d.category)));
        if (it != g_warning_classes->end()) cls = it->second.ptr();
    }
    // stacklevel 1 = the Python frame that called into the library.
    if (PyErr_WarnEx(cls, d.message.c_str(), 1) != 0) {
        // A warnings filter turned it into an error: propagate that Python
        // exception through the C++ caller and back out to Python.
        throw nb::python_error();
    }
}

}  // namespace

void bind_diagnostics(nb::module_& m) {
    // Base first, subclasses after: nanobind tries the most recently
    // registered translator first, so a subclass is matched before its base.
    nb::object base = nb::exception<Error>(m, "BallisticError", PyExc_RuntimeError);
    nb::exception<DataCoverageError>(m, "DataCoverageError", base);
    nb::exception<FrameError>(m, "FrameError", base);
    nb::exception<InputPlausibilityError>(m, "InputPlausibilityError", base);
    nb::exception<ConfigError>(m, "ConfigError", base);
    nb::exception<RemovedApiError>(m, "RemovedApiError", base);
    nb::exception<StrictDiagnosticError>(m, "StrictDiagnosticError", base);

    m.def("_install_diagnostic_handler",
        [](nb::dict classes) {
            if (!g_warning_classes) g_warning_classes = new std::unordered_map<std::string, nb::object>();
            g_warning_classes->clear();
            for (auto [k, v] : classes) {
                (*g_warning_classes)[nb::cast<std::string>(k)] = nb::borrow<nb::object>(v);
            }
            diagnostics::set_handler(python_handler);
        },
        nb::arg("classes"),
        "Internal: route library diagnostics to Python warnings; `classes` maps each "
        "category name to its warning class (called by ballistic/__init__.py).");

    m.def("_report_diagnostic",
        [](const std::string& category, const std::string& message, const std::string& key) {
            static const std::unordered_map<std::string, diagnostics::Category> names = {
                {"DataCoverage", diagnostics::Category::DataCoverage},
                {"ModelSubstitution", diagnostics::Category::ModelSubstitution},
                {"DataSkipped", diagnostics::Category::DataSkipped},
                {"ForcedStep", diagnostics::Category::ForcedStep},
                {"Extrapolation", diagnostics::Category::Extrapolation},
                {"FrameAssumption", diagnostics::Category::FrameAssumption},
                {"ConfigHint", diagnostics::Category::ConfigHint},
            };
            auto it = names.find(category);
            if (it == names.end()) throw ConfigError("unknown diagnostic category '" + category + "'");
            diagnostics::report(it->second, message, key);
        },
        nb::arg("category"), nb::arg("message"), nb::arg("key") = "",
        "Internal: report a diagnostic through the library's channel (tests and Python-side "
        "checks). category: DataCoverage, ModelSubstitution, DataSkipped, ForcedStep, "
        "Extrapolation, FrameAssumption or ConfigHint (else ConfigError); key: a non-empty "
        "key makes the note print once per process.");

    m.def("_reset_diagnostic_dedup", [] { diagnostics::Sink::instance().reset_dedup(); },
          "Internal (tests): forget which once-per-process diagnostics were already "
          "delivered.");
}
