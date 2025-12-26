# Copyright (c) 2012 Adam Karpierz
# SPDX-License-Identifier: Zlib

"""Improved dictionary access through dot notation."""

from typing import Any
from typing_extensions import Self
from collections.abc import Iterable
from collections import defaultdict

__all__ = ('adict', 'defaultadict')


class __adict:

    class __AttributeAndKeyError(AttributeError, KeyError):
        __doc__ = AttributeError.__doc__

    def __getitem__(self, key: Any) -> Any:
        """Item access"""
        try:
            return super().__getitem__(key)  # type: ignore[misc]
        except KeyError as exc:
            raise self.__AttributeAndKeyError(*exc.args) from None

    def __setitem__(self, key: Any, value: Any) -> None:
        """Item assignment"""
        try:
            super().__setitem__(key, value)  # type: ignore[misc]
        except KeyError as exc:  # pragma: no cover
            raise self.__AttributeAndKeyError(*exc.args) from None

    def __delitem__(self, key: Any) -> None:
        """Item deletion"""
        try:
            super().__delitem__(key)  # type: ignore[misc]
        except KeyError as exc:
            raise self.__AttributeAndKeyError(*exc.args) from None

    def __getattr__(self, name: str) -> Any:
        """Attribute access"""
        return self.__getitem__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Attribute assignment"""
        self.__setitem__(name, value)

    def __delattr__(self, name: str) -> None:
        """Attribute deletion"""
        self.__delitem__(name)


class adict(__adict, dict[Any, Any]):
    """Improved standard 'dict' access through dot notation."""

    def copy(self) -> Self:
        """Return a shallow copy of the adict."""
        return self.__class__(self)

    def __copy__(self) -> Self:
        """Return a shallow copy of the adict."""
        return self.copy()


class defaultadict(__adict, defaultdict[Any, Any]):
    """Improved 'collections.defaultdict' access through dot notation."""

    def copy(self) -> Self:
        """Return a shallow copy of the defaultadict."""
        return self.__class__(self.default_factory, self)

    def __copy__(self) -> Self:
        """Return a shallow copy of the defaultadict."""
        return self.copy()


del Self, Iterable, defaultdict
