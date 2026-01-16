from __future__ import annotations
import importlib.util as _util
import importlib.machinery as _mach
import os as _os
import sys as _sys
from typing import Optional as _Optional

# rmcore loader
#
# This module locates and loads the compiled rmcore extension (.so/.pyd) and then replaces itself with that module in sys.modules.
#
# Search strategy:
#   1) Look in the current package directory (rmsynth/python) for files
#      named rmcore* with an extension suffix (.so, .pyd, ...).
#   2) Look for a sibling "rmsynth" directory on each entry in sys.path
#      and search it for rmcore* binaries.
#
# The *most recently modified* candidate is loaded and installed under the name "rmsynth.rmcore".

def _find_binary_path() -> _Optional[str]:
    suffixes = tuple(_mach.EXTENSION_SUFFIXES)  # e.g. .so, .abi3.so, .pyd, etc.
    candidates: list[str] = []

    # 1) scan this package directory
    pkg_dir = _os.path.dirname(__file__)
    try:
        for name in _os.listdir(pkg_dir):
            if name.startswith("rmcore") and name.endswith(suffixes):
                candidates.append(_os.path.join(pkg_dir, name))
    except Exception:
        pass

    # 2) scan "rmsynth" subdirectories on sys.path
    for root in list(_sys.path):
        d = _os.path.join(root, "rmsynth")
        if not _os.path.isdir(d):
            continue
        try:
            for name in _os.listdir(d):
                if name.startswith("rmcore") and name.endswith(suffixes):
                    candidates.append(_os.path.join(d, name))
        except Exception:
            continue

    if not candidates:
        return None
    # deduplicate while preserving order
    uniq = list(dict.fromkeys(candidates))
    # prefer the newest binary (helps dev installs where multiple builds exist)
    uniq.sort(key=lambda p: _os.path.getmtime(p) if _os.path.exists(p) else 0, reverse=True)
    return uniq[0]

def _load_as_rmcore(path: str):
    """
    Load the compiled extension at 'path' as the module 'rmsynth.rmcore'.

    The last component of the module name must be 'rmcore' to match the
    extension's PyInit_rmcore entry point.
    """
    name = "rmsynth.rmcore"   # IMPORTANT: last component must be 'rmcore'
    loader = _mach.ExtensionFileLoader(name, path)
    spec = _util.spec_from_file_location(name, path, loader=loader)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create a loader for rmcore binary at: {path}")
    # create and initialize the extension module
    mod = _util.module_from_spec(spec)
    # execute the extension (calls PyInit_rmcore)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

# Locate and load the binary, then alias this module to the real extension.
_path = _find_binary_path()
if _path is None:
    raise ImportError(
        "Compiled rmcore extension not found on sys.path or in the package directory.\n"
        "Please build/install the project first, e.g.: `pip install -v .`"
    )

_mod = _load_as_rmcore(_path)
_sys.modules[__name__] = _mod
