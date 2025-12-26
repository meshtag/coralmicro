# Copyright (c) 1994 Adam Karpierz
# SPDX-License-Identifier: Zlib

__all__ = ('arch',)


def get_python_arch() -> str | None:
    import sys
    import platform
    machine = platform.machine().lower()
    is_32bits = (sys.maxsize <= 2**32)

    # x86
    if machine in ("amd64", "x86_64"):
        arch = "x86_64" if not is_32bits else "x86"
    elif machine in ("x86", "i386", "i686", "ia32"):
        arch = "x86"
    # ARM
    elif machine in ("aarch64",) or machine.startswith("arm"):
        arch = "arm64" if not is_32bits else "arm"
    else:
        arch = None  # unknown or unsupported

    return arch


arch = get_python_arch()
