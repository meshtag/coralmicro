# @public -- populate __all__
#
# Copyright (C) 2016 Barry Warsaw <barry@python.org>
#
# This project is licensed under the terms of the Apache 2.0 License.

from typing import overload, TypeVar, Any
from collections.abc import Callable
import sys

__all__ = ('public', 'private')

ModuleAware = TypeVar("ModuleAware", bound=Callable[..., Any])


@overload
def public(thing: ModuleAware) -> ModuleAware:
    ...
@overload
def public(**kwargs: Any) -> Any | tuple[Any]:
    ...
def public(thing: Any | None = None, **kwargs: Any) -> ModuleAware | Any | tuple[Any]:
    """Add a name or names to __all__"""
    mdict = (sys._getframe(1).f_globals  # The function call syntax.
             if thing is None else
             sys.modules[thing.__module__].__dict__)  # The decorator syntax.
    dunder_all = mdict.setdefault("__all__", [])
    if not isinstance(dunder_all, list):
        raise TypeError(f"__all__ must be a list not: {type(dunder_all)}")
    if thing is None:
        # The function call form.
        retval = []
        for key, value in kwargs.items():
            if key not in dunder_all:
                dunder_all.append(key)
            mdict[key] = value
            retval.append(value)
        return retval[0] if len(retval) == 1 else tuple(retval)
    else:
        # The decorator form.
        if kwargs:
            raise AssertionError("Keyword arguments are incompatible with use "
                                 "as decorator")
        if thing.__name__ not in dunder_all:
            dunder_all.append(thing.__name__)
    return thing


def private(thing: ModuleAware) -> ModuleAware:
    """Remove names from __all__"""
    mdict = sys.modules[thing.__module__].__dict__
    dunder_all = mdict.setdefault("__all__", [])
    if not isinstance(dunder_all, list):
        raise TypeError(f"__all__ must be a list not: {type(dunder_all)}")
    if thing.__name__ in dunder_all:
        dunder_all.remove(thing.__name__)
    return thing


del ModuleAware
del overload, TypeVar, Callable
