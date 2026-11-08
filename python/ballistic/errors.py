"""Errors and warnings raised by ballistic.

Errors (raised): every deliberate library error derives from
``BallisticError`` (a ``RuntimeError``):

- ``DataCoverageError``: required data (Earth orientation parameters, space
  weather, leap seconds, an orbit file) does not cover the requested time.
- ``FrameError``: an input is in a different reference frame than expected.
- ``InputPlausibilityError``: an input is physically implausible (units,
  below ground) and was refused.
- ``ConfigError``: a configuration value cannot be honored.
- ``RemovedApiError``: a removed API; the message names the replacement.
- ``StrictDiagnosticError``: strict mode is on (``set_strict``, or the
  environment variable ``BALLISTIC_STRICT_WARNINGS=1``) and the library
  reported an approximation that would otherwise have been a warning.

Warnings (issued, not raised): the library never substitutes a value
silently. When it continues with an approximation it issues a warning of a
``BallisticWarning`` subclass (a ``RuntimeWarning``), one per kind:

- ``DataCoverageWarning``: data near or past its edge; a value is held or
  forecast.
- ``ModelSubstitutionWarning``: a different model stands in for the
  requested one.
- ``DataSkippedWarning``: input records were skipped while loading.
- ``ForcedStepWarning``: the numerical integrator accepted a step that
  failed its own accuracy test (it could not shrink the step further).
- ``ExtrapolationWarning``: a fitted quantity is used outside the span it
  was fitted on.
- ``FrameAssumptionWarning``: an input was interpreted in an assumed frame.
- ``ConfigHintWarning``: the configuration leaves accuracy on the table
  (advice; never turned into an error).

``ballistic.set_strict(True)`` turns every ballistic warning except
``ConfigHintWarning`` into an error.
"""

import warnings as _warnings

from ._core import (  # noqa: F401  (re-exported)
    BallisticError,
    DataCoverageError,
    FrameError,
    InputPlausibilityError,
    ConfigError,
    RemovedApiError,
    StrictDiagnosticError,
)


class BallisticWarning(RuntimeWarning):
    """Base class of every warning the library issues.

    Filter it like any warning, e.g.
    ``warnings.simplefilter("ignore", ballistic.DataSkippedWarning)``.
    """


class DataCoverageWarning(BallisticWarning):
    """Data near or past its edge: a value is held, forecast or frozen."""


class ModelSubstitutionWarning(BallisticWarning):
    """A different model stands in for the requested one."""


class DataSkippedWarning(BallisticWarning):
    """Input records were skipped while loading."""


class ForcedStepWarning(BallisticWarning):
    """The integrator accepted a step that failed its error test."""


class ExtrapolationWarning(BallisticWarning):
    """A fitted quantity is used outside the span it was fitted on."""


class FrameAssumptionWarning(BallisticWarning):
    """An input was interpreted in an assumed reference frame."""


class ConfigHintWarning(BallisticWarning):
    """The configuration leaves accuracy on the table."""


# The C++ diagnostic category name (include/ballistic/core/diagnostics.hpp)
# -> the Python warning class it is issued as.
WARNING_CLASSES = {
    "DataCoverage": DataCoverageWarning,
    "ModelSubstitution": ModelSubstitutionWarning,
    "DataSkipped": DataSkippedWarning,
    "ForcedStep": ForcedStepWarning,
    "Extrapolation": ExtrapolationWarning,
    "FrameAssumption": FrameAssumptionWarning,
    "ConfigHint": ConfigHintWarning,
}


def set_strict(on: bool = True) -> None:
    """Turn every ballistic warning except ``ConfigHintWarning`` into an error (``on=True``) or back.

    Use it in pipelines that must not run on approximations. It sets
    ``warnings.simplefilter("error", BallisticWarning)`` and keeps
    ``ConfigHintWarning`` warnings; ``on=False`` restores the default
    filter for ``BallisticWarning``.

    Args:
        on: True to raise, False to warn again.
    """
    _warnings.simplefilter("error" if on else "default", BallisticWarning)
    if on:   # advice about a choice the caller made, not an approximation: never an error
        _warnings.simplefilter("default", ConfigHintWarning)
