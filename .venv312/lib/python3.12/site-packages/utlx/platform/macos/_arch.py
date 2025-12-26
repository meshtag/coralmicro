# Copyright (c) 1994 Adam Karpierz
# SPDX-License-Identifier: Zlib

__all__ = ('arch',)


def get_python_arch() -> str | None:
    import sys
    import platform
    machine = platform.machine().lower()
    is_32bits = (sys.maxsize <= 2**32)

    # x86
    if machine in ("x86_64", "amd64"):
        arch = "x86_64" if not is_32bits else "x86"
    elif machine in ("i386", "i686", "x86", "ia32"):
        arch = "x86"
    # ARM
    elif machine in ("arm64",):  # macOS never reports aarch64
        arch = "arm64" if not is_32bits else "arm"
    else:
        arch = None  # unknown or unsupported

    return arch


arch = get_python_arch()


def macos_version() -> tuple[int, ...]:
    import platform
    version  = [int(x) for x in platform.mac_ver()[0].split(".")]
    version += [0] * (3 - len(version))
    return tuple(version)
