# flake8-in-file-ignores: noqa: A005,F401,F403,F821

# Copyright (c) 2004 Adam Karpierz
# SPDX-License-Identifier: Zlib

from ._c_ptr import * ; del _c_ptr  # type: ignore[name-defined]
