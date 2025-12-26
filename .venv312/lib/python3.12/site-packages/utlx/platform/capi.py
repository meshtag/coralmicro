# flake8-in-file-ignores: noqa: A005,F401,F403,F405

# Copyright (c) 1994 Adam Karpierz
# SPDX-License-Identifier: Zlib

from ._detect import is_windows, is_linux, is_macos

if is_windows:  # pragma: no cover
    from .windows.capi import *  # type: ignore[assignment, unused-ignore]
elif is_linux:  # pragma: no cover
    from .linux.capi import *    # type: ignore[assignment, unused-ignore]
elif is_macos:  # pragma: no cover
    from .macos.capi import *    # type: ignore[assignment, unused-ignore]

del is_windows, is_linux, is_macos
