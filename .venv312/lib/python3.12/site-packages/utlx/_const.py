# Copyright (c) 2012 Adam Karpierz
# SPDX-License-Identifier: Zlib

from typing import Any
import weakref

__all__ = ('const', 'weakconst')


class const:

    __slots__ = ('__value', '__doc__')

    def __init__(self, value: Any = None,
                 doc: str | None = None) -> None:
        """Initializer"""
        self.__value = value
        self.__doc__ = doc

    def __get__(self, this: Any, cls: Any) -> Any:
        """Access handler"""
        return self.__value

    def __set__(self, this: Any, value: Any) -> None:
        """Assignment handler"""
        raise TypeError("readonly attribute")

    def __delete__(self, this: Any) -> None:
        """Deletion handler"""
        raise TypeError("readonly attribute")


class weakconst:

    __slots__ = ('__value', '__doc__')

    def __init__(self, value: Any = None,
                 doc: str | None = None) -> None:
        """Initializer"""
        self.__value = weakref.ref(value)
        self.__doc__ = doc

    def __get__(self, this: Any, cls: Any) -> Any:
        """Access handler"""
        return self.__value()

    def __set__(self, this: Any, value: Any) -> None:
        """Assignment handler"""
        raise TypeError("readonly attribute")

    def __delete__(self, this: Any) -> None:
        """Deletion handler"""
        raise TypeError("readonly attribute")
