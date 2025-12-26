# Copyright (c) 1994 Adam Karpierz
# SPDX-License-Identifier: Zlib

__all__ = ('arch', 'capi', 'winapi')

from ._arch import arch
from . import capi
from . import winapi
