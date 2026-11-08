"""The NRLMSIS 2.1 atmosphere and HWM14 wind models, installed on request.

Model-based drag (``make_drag_environment``, ``recommended_config`` with drag,
the ``ballistic.drag.from_models_*`` tables) needs NRLMSIS 2.1 and HWM14. The
package does not contain them: NRLMSIS 2.1 is licensed by the U.S. Naval
Research Laboratory for research, academic and non-profit use only
(NRL-SOF-014-1), so the two models are a separate plugin that a user installs
knowingly, once per machine:

    >>> import ballistic
    >>> ballistic.install_nrl_models(accept_license=True)

The plugin is a small shared library (``ballistic_nrl.so``, ``.dylib`` or
``.dll``) downloaded from the project's GitHub release for this version into
``<data directory>/nrl/``, where the library finds it in every later session.
On a machine without internet access, download the file for your platform
from the release page (``nrl_models().download_url``) elsewhere and install it
with ``install_nrl_models(accept_license=True, source="path/to/file")``, or
copy it to ``nrl_models().install_path`` yourself.

Without the plugin everything else works: propagation without drag, and drag
from measured density (``ballistic.drag.from_accelerometer_density``).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import os
import platform
import ssl
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import _core

RELEASES_URL = "https://github.com/nvhuynh16/ballistic/releases"
LICENSE_URL = ("https://github.com/nvhuynh16/ballistic/blob/main/docs/licenses/"
               "nrlmsis2.1_license.txt")

LICENSE_NOTICE = f"""\
NRLMSIS 2.1 and HWM14 for ballistic (model-based atmospheric drag)

NRLMSIS 2.1 is licensed by the U.S. Naval Research Laboratory under the Open
Source Academic Research License Agreement NRL-SOF-014-1:

  * use is permitted for research, academic and non-profit purposes only;
  * COMMERCIAL USE IS NOT PERMITTED (for commercial use, obtain a license
    from NRL, or build drag from measured density with
    ballistic.drag.from_accelerometer_density, which needs no NRL code);
  * copies and derivative works must carry the license agreement.

Full text: {LICENSE_URL}
HWM14 (Drob et al. 2015) is distributed as a U.S. Government work
(17 U.S.C. 105; the project's reading, NRL attached no license).
"""


def _platform_tag() -> tuple[str, str]:
    """(release asset tag, file extension) for this machine."""
    machine = platform.machine().lower()
    arch = {"amd64": "x86_64", "x86_64": "x86_64", "arm64": "arm64",
            "aarch64": "arm64"}.get(machine, machine)
    if sys.platform.startswith("linux"):
        return f"linux-{arch}", "so"
    if sys.platform == "darwin":
        return f"macos-{arch}", "dylib"
    if sys.platform == "win32":
        return f"windows-{arch}", "dll"
    return f"{sys.platform}-{arch}", "so"


def _asset_name() -> str:
    tag, ext = _platform_tag()
    return f"ballistic_nrl-{tag}.{ext}"


def _release_url(name: str) -> str:
    return f"{RELEASES_URL}/download/v{_core.__version__}/{name}"


@dataclass(frozen=True)
class NrlModelsStatus:
    """Whether the NRL models plugin is installed, and where to get it.

    Attributes:
        installed: True when the plugin is loaded (model-based drag works).
        path: the loaded plugin file; empty when not installed.
        models: the models it provides, e.g. "NRLMSIS 2.1, HWM14".
        problem: why a plugin that was found could not be loaded; empty if none.
        install_path: where install_nrl_models puts it (and where it is found).
        download_url: the plugin for this platform and version on GitHub.
    """

    installed: bool
    path: str
    models: str
    problem: str
    install_path: str
    download_url: str

    def __bool__(self) -> bool:
        return self.installed


def nrl_models() -> NrlModelsStatus:
    """Report whether the NRLMSIS 2.1 / HWM14 plugin is installed.

    Looks for it the way model-based drag does (the environment variable
    ``BALLISTIC_NRL_MODELS``, then ``<data directory>/nrl/``) and loads it if
    found. True in a boolean context when installed.
    """
    d = _core._nrl_models_status()
    return NrlModelsStatus(
        installed=bool(d["loaded"]), path=d["path"], models=d["models"],
        problem=d["problem"], install_path=d["install_path"],
        download_url=_release_url(_asset_name()))


def _download(url: str, timeout_s: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "ballistic-python"})
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(request, timeout=timeout_s, context=ctx) as response:
        return response.read()


def _expected_sha256(asset: str, timeout_s: float) -> str:
    """The SHA-256 of `asset` listed in the release's SHA256SUMS file."""
    text = _download(_release_url("SHA256SUMS"), timeout_s).decode("utf-8")
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("*") == asset:
            return parts[0].lower()
    raise _core.ConfigError(
        f"install_nrl_models: the release v{_core.__version__} lists no {asset} in its "
        f"SHA256SUMS ({_release_url('SHA256SUMS')}): no plugin is published for this "
        "platform. Build it from the source tree (CMake target ballistic_nrl) and install "
        "it with source=.")


def install_nrl_models(accept_license: bool = False, *, source: str | os.PathLike | None = None,
                       timeout_s: float = 60.0) -> Path | None:
    """Install the NRLMSIS 2.1 atmosphere and HWM14 wind models (model-based drag).

    NRLMSIS 2.1 is licensed for research, academic and non-profit use only.
    Without ``accept_license=True`` this prints the license terms and
    installs nothing. With it, the plugin is downloaded from the GitHub
    release of this version (checked against the release's SHA-256 list), or
    copied from ``source``, into ``<data directory>/nrl/``, loaded, and found
    automatically in every later session. Nothing is shown when drag is used
    afterwards.

    Args:
        accept_license: True to accept the NRLMSIS 2.1 license terms
            (NRL-SOF-014-1, non-commercial use only) and install.
        source: a plugin file already downloaded (offline installation), for
            example ``ballistic_nrl-linux-x86_64.so`` from the release page or
            ``build/nrl/ballistic_nrl.so`` from a source build. None downloads
            it (not allowed with ``BALLISTIC_OFFLINE=1``).
        timeout_s: network timeout for the download [s].

    Returns:
        The installed plugin's path, or None when the license was not accepted.

    Raises:
        ConfigError: the download failed or was refused offline, the file
            does not match the published checksum, or it is not a plugin this
            version can load.
    """
    status = nrl_models()
    if not accept_license:
        print(LICENSE_NOTICE)
        print("Nothing was installed. To accept these terms and install the models, run\n"
              "    ballistic.install_nrl_models(accept_license=True)\n"
              f"(offline: download {_asset_name()} from {RELEASES_URL} and pass source=).")
        return None
    if status.installed and source is None:
        return Path(status.path)

    target = Path(status.install_path)
    if source is not None:
        src = Path(source)
        if not src.is_file():
            raise _core.ConfigError(f"install_nrl_models: no file at {src}")
        data = src.read_bytes()
        origin = str(src)
    else:
        if os.environ.get("BALLISTIC_OFFLINE") == "1":
            raise _core.ConfigError(
                "install_nrl_models: BALLISTIC_OFFLINE=1 forbids the download. Download "
                f"{_asset_name()} from {status.download_url} on another machine and run "
                "install_nrl_models(accept_license=True, source='path/to/that/file').")
        asset = _asset_name()
        url = _release_url(asset)
        try:
            expected = _expected_sha256(asset, timeout_s)
            data = _download(url, timeout_s)
        except (urllib.error.URLError, OSError) as e:
            raise _core.ConfigError(
                f"install_nrl_models: could not download {url} ({e}). Offline: download it "
                "on another machine and pass source='path/to/file'.") from e
        got = hashlib.sha256(data).hexdigest()
        if got != expected:
            raise _core.ConfigError(
                f"install_nrl_models: {url} has SHA-256 {got}, the release lists {expected}; "
                "nothing was installed (a damaged or altered download). Try again.")
        origin = url

    if status.installed:
        raise _core.ConfigError(
            f"install_nrl_models: the models are already loaded in this process from "
            f"{status.path}; restart Python and install again to replace them.")

    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    part.write_bytes(data)
    part.replace(target)
    try:
        _core._load_nrl_models(str(target))
    except _core.ConfigError:
        target.unlink(missing_ok=True)
        raise
    record = target.with_name("ACCEPTED.txt")
    record.write_text(
        f"License accepted on {_dt.datetime.now(_dt.timezone.utc):%Y-%m-%d %H:%M} UTC "
        f"through ballistic.install_nrl_models (ballistic {_core.__version__}).\n"
        f"Plugin: {target.name}, SHA-256 {hashlib.sha256(data).hexdigest()}, from {origin}\n\n"
        + LICENSE_NOTICE, encoding="utf-8")
    return target

