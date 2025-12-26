# Copyright (c) 1994 Adam Karpierz
# SPDX-License-Identifier: Zlib

import ctypes as ct

from ._detect import is_cpython

__all__ = ('from_oid',)

if is_cpython:

    def from_oid(oid: int | None) -> object | None:
        return ct.cast(oid, ct.py_object).value if oid else None

else:

    def from_oid(oid: int | None) -> object | None:
        from platform import python_implementation
        raise NotImplementedError("from_oid() currently works only on CPython!\n"
                                  f"Current interpreter: {python_implementation()}")
