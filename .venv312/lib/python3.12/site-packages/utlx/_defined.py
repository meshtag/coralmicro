# Copyright (c) 2012 Adam Karpierz
# SPDX-License-Identifier: Zlib

import sys

__all__ = ('defined',)


def defined(varname: str,  # type: ignore[no-untyped-def]
            __getframe=sys._getframe) -> bool:
    frame = __getframe(1)
    return varname in frame.f_locals or varname in frame.f_globals


del sys
