# Alex Martelli's 'Borg' singleton implementation:
# http://python-3-patterns-idioms-test.readthedocs.io/en/latest/Singleton.html#id1
#
# Modified (and a bit improved/enhanced) by Adam Karpierz, 2014

from typing import Any
from typing_extensions import Self

__all__ = ('Borg',)


class Borg:

    __shared_state: dict[str, Any] = {}

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        """Constructor"""
        self = super().__new__(cls)
        self.__dict__ = Borg.__shared_state
        return self
