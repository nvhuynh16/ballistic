"""build_info(): how the extension was compiled.

``ballistic._core.build_info()`` says how the extension was compiled;
``ballistic.doctor()`` prints it. (The NRL atmosphere and wind models are a
run-time plugin, not a build property: test_nrl_models.py.)
"""

import ballistic


def test_keys_and_types():
    info = ballistic._core.build_info()
    assert set(info) == {"with_openmp", "compiler", "cpp_standard", "build_type"}
    assert isinstance(info["with_openmp"], bool)
    assert isinstance(info["compiler"], str) and info["compiler"]
    assert info["cpp_standard"] >= 23
    assert info["build_type"] in ("Release", "Debug")


def test_doctor_prints_the_build_line(capsys):
    report = ballistic.doctor(load=False, self_test=False, verbose=True)
    out = capsys.readouterr().out
    assert ballistic._core.build_info()["compiler"] in out
    names = [c.name for c in report.checks]
    assert "build" in names and "NRL models (drag)" in names
