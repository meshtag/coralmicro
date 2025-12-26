# Copyright (c) 1994 Adam Karpierz
# SPDX-License-Identifier: Zlib

__all__ = ('arch', 'macos_version', 'capi')

from ._arch import arch, macos_version
from . import capi
