"""Choose which ballistic package a script imports.

A source checkout's python/ directory holds the package, but it imports only
after an in-place build has put the compiled _core next to it. Without that
build, putting python/ first on sys.path would hide the installed package
(pip install ballistic-orbit) behind a copy that cannot import.

    import _source_tree
    _source_tree.use_source_tree()
"""

import importlib.machinery
import os
import sys

PYTHON_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))


def has_built_core(python_dir=PYTHON_DIR):
    """True when python_dir/ballistic holds a compiled _core extension."""
    package = os.path.join(python_dir, 'ballistic')
    return any(os.path.isfile(os.path.join(package, '_core' + suffix))
               for suffix in importlib.machinery.EXTENSION_SUFFIXES)


def use_source_tree():
    """Put the checkout's python/ first on sys.path if it is built in place;
    otherwise leave sys.path alone, so the installed package is used."""
    if has_built_core() and PYTHON_DIR not in sys.path:
        sys.path.insert(0, PYTHON_DIR)
