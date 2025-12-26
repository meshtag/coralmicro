# Read & write properties
#
# Copyright (c) 2006 by Philipp "philiKON" von Weitershausen
#                       philikon@philikon.de
#
# Freely distributable under the terms of the Zope Public License, v2.1.
#
# See _rwproperty.txt for detailed explanations

from typing import TypeAlias, Any
from collections.abc import Callable
import sys

__all__ = ('getproperty', 'setproperty', 'delproperty')

AnyCallable: TypeAlias = Callable[..., Any]


class _property:

    def __new__(cls, func: AnyCallable) -> property:  # type: ignore[misc]
        """Constructor"""
        # ugly, but common hack
        frame = sys._getframe(1)
        oldprop = frame.f_locals.get(func.__name__)
        if oldprop is None:
            return cls.create_property(func)
        elif isinstance(oldprop, property):
            return cls.enhance_property(oldprop, func)
        else:
            raise TypeError("read & write properties cannot be mixed with "
                            "other attributes except regular property objects.")

    @staticmethod
    def create_property(func: AnyCallable) -> property:
        raise NotImplementedError()

    @staticmethod
    def enhance_property(oldprop: property, func: AnyCallable) -> property:
        raise NotImplementedError()


class getproperty(_property):

    @staticmethod
    def create_property(func: AnyCallable) -> property:
        return property(func)

    @staticmethod
    def enhance_property(oldprop: property, func: AnyCallable) -> property:
        return property(func, oldprop.fset, oldprop.fdel)


class setproperty(_property):

    @staticmethod
    def create_property(func: AnyCallable) -> property:
        return property(None, func)

    @staticmethod
    def enhance_property(oldprop: property, func: AnyCallable) -> property:
        return property(oldprop.fget, func, oldprop.fdel)


class delproperty(_property):

    @staticmethod
    def create_property(func: AnyCallable) -> property:
        return property(None, None, func)

    @staticmethod
    def enhance_property(oldprop: property, func: AnyCallable) -> property:
        return property(oldprop.fget, oldprop.fset, func)


if __name__ == "__main__":
    import doctest
    doctest.testfile("_rwproperty.txt")
