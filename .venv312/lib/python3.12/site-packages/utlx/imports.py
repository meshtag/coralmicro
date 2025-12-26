# Copyright (c) 2012 Adam Karpierz
# SPDX-License-Identifier: Zlib

from typing import TypeAlias
from os import PathLike
import sys
import types
import importlib.util
from pathlib import Path

__all__ = ('import_static', 'import_file')

StrPath: TypeAlias = str | PathLike[str]


def import_static(name: str, *, package: str | None = None,
                  reload: bool = False) -> types.ModuleType:
    """
    Import a module from the standard library or installed packages, \
    bypassing local shadowing and optionally forcing reload.

    Args:
        name (str): Fully qualified module name.
        package (str | None): Optional relative import context.
        reload (bool): If True, forces reloading the module.

    Returns:
        types.ModuleType: Imported module object.

    Raises:
        ImportError: If the module cannot be found or loaded.
    """
    spec = importlib.util.find_spec(name, package=package)
    if spec is None or spec.loader is None:
        raise ImportError(f"Module '{name}' not found")

    if not reload and spec.name in sys.modules:
        return sys.modules[spec.name]

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def import_file(path: StrPath, *,
                name: str | None = None, reload: bool = False,
                strict_sys_path: bool = True) -> types.ModuleType:
    """
    Import a module directly from a file path or package directory.

    Args:
        path (str | PathLike[str]): Path to .py file or package directory.
        name (str | None): Optional name to assign to the module.
        reload (bool): If True, forces reloading the module.
        strict_sys_path (bool): If True, ensures the file is within sys.path.

    Returns:
        types.ModuleType: Imported module object.

    Raises:
        ImportError: If the file does not exist, cannot be loaded,
                     or is outside sys.path when strict_sys_path is True.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise ImportError(f"No module named '{path.stem}'")

    if strict_sys_path:
        if not any(path.is_relative_to(Path(p).resolve())
                   for p in sys.path if p):
            raise ImportError(f"Module path '{path}' is not within sys.path")

    if name is None:  name = path.stem
    if path.is_dir(): path = path/"__init__.py"

    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module named '{name}'")

    if not reload and spec.name in sys.modules:
        return sys.modules[spec.name]

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


del StrPath, PathLike
