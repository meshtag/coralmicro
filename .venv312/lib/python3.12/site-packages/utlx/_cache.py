# Copyright (c) 2012 Adam Karpierz
# SPDX-License-Identifier: Zlib

from typing import TypeVar, TypeAlias, Any
from collections.abc import Callable

__all__ = ('cached', 'cached_property')

_P = TypeVar("_P", bound=object)
_T = TypeVar("_T")

AnyCallable: TypeAlias = Callable[..., Any]


def cached(method: AnyCallable) -> AnyCallable:
    """Decorator to simple cache method's result"""

    from functools import wraps

    @wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        key: int = hash(method)
        try:
            cache = self.__cache__
        except AttributeError:
            cache = self.__cache__ = {}
        try:
            result = cache[key]
        except KeyError:
            result = cache[key] = method(self, *args, **kwargs)
        return result

    return wrapper


def cached_property(fget: Callable[[_P], _T] | None = None,
                    fset: Callable[[_P, _T], None] | None = None,
                    fdel: Callable[[_P], None] | None = None,
                    doc: str | None = None) -> property:
    """Decorator to simple cache property attribute.

    fget
      function to be used for cached getting an attribute value
    fset
      function to be used for setting an attribute value
    fdel
      function to be used for deleting an attribute
    doc
      docstring
    """
    from functools import wraps

    _fget, _fset, _fdel = fget, fset, fdel
    if fget is not None:
        key: int = hash(fget)

        @wraps(fget)
        def _fget(self: _P) -> _T:  # noqa: F811
            try:
                cache = self.__cache__  # type: ignore[attr-defined]
            except AttributeError:
                cache = self.__cache__ = {}  # type: ignore[attr-defined]
            result: _T
            try:
                result = cache[key]
            except KeyError:
                result = cache[key] = fget(self)
            return result

        if fset is not None:
            @wraps(fset)
            def _fset(self: _P, value: _T) -> None:  # noqa: F811
                try:
                    del self.__cache__[key]  # type: ignore[attr-defined]
                except (AttributeError, KeyError):
                    pass
                fset(self, value)

        if fdel is not None:
            @wraps(fdel)
            def _fdel(self: _P) -> None:  # noqa: F811
                try:
                    del self.__cache__[key]  # type: ignore[attr-defined]
                except (AttributeError, KeyError):
                    pass
                fdel(self)

    return property(_fget, _fset, _fdel, doc)


del TypeVar, TypeAlias, Callable, AnyCallable
