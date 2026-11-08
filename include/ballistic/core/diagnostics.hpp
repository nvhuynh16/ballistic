#pragma once

// =============================================================================
// Diagnostics channel: one place every non-fatal warning goes through
// =============================================================================
//
// The library never substitutes a value silently. When it continues with an
// approximation (a held Earth-orientation value, forecast space weather, a
// frozen drag table, a forced integrator step), it reports a diagnostic here.
//
//   - C++ default: printed to stderr ("ballistic [Category]: message"); a
//     keyed diagnostic only the first time (see below).
//   - Python: the extension installs a handler that issues a Python warning
//     of the matching ballistic.*Warning class, so warnings filters apply
//     (and "error" filters turn them into exceptions).
//   - Strict mode (BALLISTIC_STRICT_WARNINGS=1 in the environment, or
//     set_strict(true)): every diagnostic except ConfigHint throws
//     StrictDiagnosticError instead (a ConfigHint is advice about a choice
//     the caller made, not an approximation of data; it is still printed).
//     Use it in tests and pipelines that must not run on approximations.
//     When a handler is installed (Python), the handler decides: Python's
//     strict mode is a warnings filter ("error").
//
// report() deduplicates when given a non-empty key: the same (category, key)
// is delivered once per process. Callers that dedup themselves (Trajectory
// records each note once per trajectory) pass an empty key.
// =============================================================================

#include "errors.hpp"

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <utility>

namespace ballistic {
namespace diagnostics {

/// What kind of approximation a diagnostic reports (Python: one ballistic.*Warning class each).
enum class Category {
    DataCoverage,       // data near or past its edge; a value is held or forecast
    ModelSubstitution,  // a different model stands in for the requested one
    DataSkipped,        // input records were skipped while loading
    ForcedStep,         // the integrator accepted a step that failed its error test
    Extrapolation,      // a fitted quantity is used outside the span it was fitted on
    FrameAssumption,    // an input was interpreted in an assumed frame
    ConfigHint,         // the configuration leaves accuracy on the table
};

/// The category's name as text, e.g. "DataCoverage".
[[nodiscard]] inline constexpr std::string_view category_name(Category c) noexcept {
    switch (c) {
        case Category::DataCoverage: return "DataCoverage";
        case Category::ModelSubstitution: return "ModelSubstitution";
        case Category::DataSkipped: return "DataSkipped";
        case Category::ForcedStep: return "ForcedStep";
        case Category::Extrapolation: return "Extrapolation";
        case Category::FrameAssumption: return "FrameAssumption";
        case Category::ConfigHint: return "ConfigHint";
    }
    return "Unknown";
}

/// True for the categories that strict mode never turns into an error: advice
/// (ConfigHint) rather than an approximation of data.
[[nodiscard]] inline constexpr bool is_advisory(Category c) noexcept {
    return c == Category::ConfigHint;
}

/// One delivered diagnostic, as passed to a Handler.
struct Diagnostic {
    Category category;
    std::string key;      // dedup key; empty = caller deduplicated
    std::string message;
};

/// A function that receives every diagnostic instead of the stderr default.
using Handler = std::function<void(const Diagnostic&)>;

/// The process-wide diagnostics channel (a singleton; the free functions below forward to it).
class Sink {
public:
    /// The shared instance. Strict mode starts on if BALLISTIC_STRICT_WARNINGS is set to a
    /// value that is not empty and does not start with '0'.
    static Sink& instance() {
        static Sink sink;
        return sink;
    }

    /// Deliver a diagnostic of `category` with text `message`. A non-empty `key` makes it
    /// once-per-process: a later report with the same (category, key) is dropped. Throws
    /// StrictDiagnosticError in strict mode when no handler is installed (never for
    /// ConfigHint); with a handler installed, whatever the handler throws propagates.
    /// Thread-safe. The handler runs outside the lock, so it may itself call into the library.
    void report(Category category, std::string message, std::string key = {}) {
        Handler handler;
        bool strict = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!key.empty()) {
                const std::string k = std::string(category_name(category)) + "|" + key;
                if (!seen_.insert(k).second) return;
            }
            handler = handler_;
            strict = strict_;
        }
        Diagnostic d{category, std::move(key), std::move(message)};
        if (handler) {
            // An installed handler owns strictness (Python: warnings filters).
            handler(d);
        } else if (strict && !is_advisory(d.category)) {
            throw StrictDiagnosticError(std::string(category_name(d.category)) + ": " + d.message);
        } else {
            std::fprintf(stderr, "ballistic [%.*s]: %s\n",
                         static_cast<int>(category_name(d.category).size()),
                         category_name(d.category).data(), d.message.c_str());
        }
    }

    /// Install `h` to receive every diagnostic (it then also owns strictness); an empty
    /// handler (nullptr) restores the stderr default.
    void set_handler(Handler h) {
        std::lock_guard<std::mutex> lock(mutex_);
        handler_ = std::move(h);
    }

    /// Turn strict mode on or off (applies only while no handler is installed).
    void set_strict(bool on) {
        std::lock_guard<std::mutex> lock(mutex_);
        strict_ = on;
    }

    /// True if strict mode is on.
    [[nodiscard]] bool strict() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return strict_;
    }

    /// Forget which keyed diagnostics were delivered, so they print again (for tests).
    void reset_dedup() {
        std::lock_guard<std::mutex> lock(mutex_);
        seen_.clear();
    }

private:
    Sink() {
        const char* env = std::getenv("BALLISTIC_STRICT_WARNINGS");
        strict_ = env && env[0] != '\0' && env[0] != '0';
    }

    mutable std::mutex mutex_;
    Handler handler_;
    bool strict_ = false;
    std::set<std::string> seen_;
};

/// Shorthands for Sink::instance().report / set_handler / set_strict / strict.
inline void report(Category category, std::string message, std::string key = {}) {
    Sink::instance().report(category, std::move(message), std::move(key));
}

inline void set_handler(Handler h) { Sink::instance().set_handler(std::move(h)); }
inline void set_strict(bool on) { Sink::instance().set_strict(on); }
[[nodiscard]] inline bool strict() { return Sink::instance().strict(); }

}  // namespace diagnostics
}  // namespace ballistic
