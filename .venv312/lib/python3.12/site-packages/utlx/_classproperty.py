# Copyright (c) 2012 Adam Karpierz
# SPDX-License-Identifier: Zlib

import typing
from typing import TypeVar, TypeAlias, Generic, Any
from typing_extensions import Self
from collections.abc import Callable

__all__ = ('classproperty',)

_T = TypeVar("_T")
_C = TypeVar("_C", bound=type)


class classproperty(Generic[_C, _T]):

    __slots__ = ('_fget', 'fset', 'fdel', '__doc__')

    def __init__(self, fget: Callable[[_C], _T] | None = None,
                 fset: Any = None, fdel: Any = None,
                 doc: str | None = None) -> None:
        """Initializer"""
        self._fget = fget
        if fset is not None or fdel is not None:
            raise ValueError("classproperty only implements fget.")
        self.fset = fset
        self.fdel = fdel
        if doc is None and self._fget is not None:
            doc = self._fget.__doc__
        self.__doc__ = doc

    def __get__(self, this: Any, cls: _C | None = None) -> _T:
        """Access handler"""
        fget = self._fget
        if fget is None:
            raise AttributeError("unreadable attribute")
        if not callable(fget):
            raise TypeError("'{}' object is not callable".format(
                            type(fget).__name__))
        return fget(cls or typing.cast(_C, type(this)))

    @property
    def fget(self) -> Callable[[_C], _T] | None:
        return self._fget

    def getter(self, fget: Callable[[_C], _T] | None = None) -> Self:
        self._fget = fget
        return self


del TypeVar, TypeAlias, Generic, Self, Callable
