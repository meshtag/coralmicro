# Copyright (c) 2012 Adam Karpierz
# SPDX-License-Identifier: Zlib

import typing
from typing import Any
from collections.abc import Iterable, Sequence
try:
    from System import String  # type: ignore[import-not-found]
except ImportError:
    String = str

__all__ = ('issubtype', 'isiterable', 'issequence', 'remove_all',
           'print_refinfo')


def issubtype(x: Any, t: Any) -> bool:
    return isinstance(x, type) and issubclass(x, t)


def isiterable(x: Any) -> bool:
    return (isinstance(x, (Iterable, typing.Iterable))
            and not isinstance(x, (bytes, str, String)))


def issequence(x: Any) -> bool:
    return (isinstance(x, (Sequence, typing.Sequence))
            and not isinstance(x, (bytes, str, String)))


def remove_all(seq: list[Any], value: Any) -> None:
    seq[:] = (item for item in seq if item != value)


def print_refinfo(obj: Any) -> None:
    import sys
    type_name = getattr(type(obj), "__name__", getattr(
                getattr(obj, "__class__", None), "__name__",
                "???"))
    ref_count = ((sys.getrefcount(obj) - 2)
                 if hasattr(sys, "getrefcount") else None)
    print("Object info report",            file=sys.stderr)
    print("    obj type: ", type_name,     file=sys.stderr)
    print("    obj id:   ", id(obj),       file=sys.stderr)
    if ref_count is not None:
        print("    ref count:", ref_count, file=sys.stderr)
