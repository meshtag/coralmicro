# flake8-in-file-ignores: noqa: F821

# Copyright (c) 2011 Adam Karpierz
# SPDX-License-Identifier: Zlib

# Strictly based on Thomas Heller's recipe.
# https://sourceforge.net/p/ctypes/mailman/message/22650014/

"""This module implements pointer arithmetic for ctypes pointers."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import TypeAlias, SupportsIndex, Any
import ctypes as ct

__all__ = ('c_ptr_add', 'c_ptr_sub', 'c_ptr_iadd', 'c_ptr_isub', 'POINTER')

if TYPE_CHECKING:
    Pointer: TypeAlias = ct._Pointer


def c_ptr_add(ptr: Pointer[Any], other: SupportsIndex) -> Pointer[Any]:
    """
    Add an integer to a pointer instance.

    Returns a new pointer:

    >>> import ctypes
    >>> string_ptr = ctypes.c_char_p(b'foobar')
    >>> string_ptr.value
    b'foobar'
    >>> p2 = c_ptr_add(string_ptr, 3)
    >>> print(p2.value)
    b'bar'
    >>> string_ptr.value
    b'foobar'
    >>>
    """
    try:
        offset = other.__index__()
    except AttributeError:
        raise TypeError("Can only add integer to pointer")
    void_p = ct.cast(ptr, ct.c_void_p)
    void_p.value = (void_p.value or 0) + offset  # * ct.sizeof(ptr._type_)
    return ct.cast(void_p, type(ptr))


def c_ptr_sub(ptr: Pointer[Any], other: Pointer[Any] | SupportsIndex) -> Pointer[Any] | int:
    """
    Substract an integer or a pointer from a pointer.

    Returns a new pointer or an integer.

    >>> import ctypes
    >>> string_ptr = ctypes.c_char_p(b'foobar')
    >>> string_ptr.value
    b'foobar'
    >>> p2 = c_ptr_add(string_ptr, 3)
    >>> print(p2.value)
    b'bar'
    >>> string_ptr.value
    b'foobar'
    >>> print(c_ptr_sub(p2, string_ptr))
    3
    >>> print(c_ptr_sub(string_ptr, p2))
    -3
    >>>
    >>> p3 = c_ptr_sub(p2, 3)
    >>> p3.value
    b'foobar'
    >>>
    >>> c_ptr_sub(string_ptr, p3)
    0
    >>>
    """
    if isinstance(other, (ct._Pointer, ct.c_void_p, ct.c_char_p, ct.c_wchar_p)):
        if type(ptr) is not type(other):
            raise TypeError("Both pointers must be of the same type")
        return ((ct.cast(ptr,   ct.c_void_p).value or 0) -  # noqa: W504
                (ct.cast(other, ct.c_void_p).value or 0))
    else:
        try:
            offset = other.__index__()
        except AttributeError:
            raise TypeError("Can only substract pointer or integer from pointer")
        void_p = ct.cast(ptr, ct.c_void_p)
        void_p.value = (void_p.value or 0) - offset  # * ct.sizeof(ptr._type_)
        return ct.cast(void_p, type(ptr))


def c_ptr_iadd(ptr: Pointer[Any], other: SupportsIndex) -> None:
    """
    Add an integer to a pointer instance in place:

    >>> import ctypes
    >>> string_ptr = ctypes.c_char_p(b'foobar')
    >>> string_ptr.value
    b'foobar'
    >>> c_ptr_iadd(string_ptr, 3)
    >>> string_ptr.value
    b'bar'
    >>>
    """
    try:
        offset = other.__index__()
    except AttributeError:
        raise TypeError("Can only add integer to pointer")
    void_p = ct.cast(ct.pointer(ptr), ct.POINTER(ct.c_void_p)).contents
    void_p.value = (void_p.value or 0) + offset  # * ct.sizeof(ptr._type_)


def c_ptr_isub(ptr: Pointer[Any], other: SupportsIndex) -> None:
    """
    Substract an integer or a pointer from a pointer.

    Returns a new pointer or an integer.

    >>> import ctypes
    >>> string_ptr = ctypes.c_char_p(b'foobar')
    >>> string_ptr.value
    b'foobar'
    >>> c_ptr_iadd(string_ptr, 4)
    >>> string_ptr.value
    b'ar'
    >>> c_ptr_isub(string_ptr, 2)
    >>> string_ptr.value
    b'obar'
    >>> c_ptr_isub(string_ptr, 1)
    >>> string_ptr.value
    b'oobar'
    >>> c_ptr_isub(string_ptr, 1)
    >>> string_ptr.value
    b'foobar'
    >>>
    """
    try:
        offset = other.__index__()
    except AttributeError:
        raise TypeError("Can only substract integer from pointer")
    void_p = ct.cast(ct.pointer(ptr), ct.POINTER(ct.c_void_p)).contents
    void_p.value = (void_p.value or 0) - offset  # * ct.sizeof(ptr._type_)


if TYPE_CHECKING:
    POINTER: TypeAlias = Pointer
    del Pointer
else:
    POINTER = ct.POINTER
del TypeAlias, SupportsIndex, Any


if __name__ == "__main__":
    import doctest
    doctest.testmod()
