#pragma once

// =============================================================================
// Error taxonomy
// =============================================================================
//
// Every error the library raises on purpose derives from ballistic::Error,
// which derives from std::runtime_error (so existing catch sites still work).
// The subclasses say what kind of problem it is; Python sees the same
// hierarchy (ballistic.BallisticError and subclasses, which derive from
// RuntimeError).
//
// Message style: what happened, how large the consequence is, and what to do.
//   "drag has no space weather past 2026-03-01 00:00:00 UTC: the loaded file  // guard-exempt(PUB-DATE): data
//    ends there. Propagation stopped; the state is kept. Load a newer SW-All.csv."
// =============================================================================

#include <stdexcept>
#include <string>

namespace ballistic {

/// Base of every error the library throws on purpose. Catch this to catch them all.
class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/// Required data does not cover the request: Earth orientation parameters, space weather,
/// leap seconds, a planetary ephemeris, an orbit file. The message names the file to load.
class DataCoverageError : public Error {
public:
    using Error::Error;
};

/// A state, vector or file is in a different reference frame than the call expects (for
/// example an inertial velocity given as Earth-fixed), or its frame cannot be determined.
class FrameError : public Error {
public:
    using Error::Error;
};

/// An input is physically implausible (wrong units, below ground, unbound when an orbit was
/// expected, a malformed record) and was refused rather than used.
class InputPlausibilityError : public Error {
public:
    using Error::Error;
};

/// A configuration value cannot be honored (e.g. a gravity degree above the loaded model)
/// or is missing.
class ConfigError : public Error {
public:
    using Error::Error;
};

/// A function or option that was removed was called; the message names the replacement.
class RemovedApiError : public Error {
public:
    using Error::Error;
};

/// A warning (see diagnostics.hpp) turned into an error because strict mode is on
/// (environment BALLISTIC_STRICT_WARNINGS=1 or diagnostics::set_strict(true)).
class StrictDiagnosticError : public Error {
public:
    using Error::Error;
};

}  // namespace ballistic
