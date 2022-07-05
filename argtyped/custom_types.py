import enum
from typing import Any, List, Optional, Tuple, Type, TypeVar, Union

import typing_inspect

__all__ = [
    "Enum",
    "auto",  # also export auto for convenience
    "Switch",
    "is_choices",
    "is_enum",
    "is_list",
    "is_optional",
    "unwrap_choices",
    "unwrap_list",
    "unwrap_optional",
]

auto = enum.auto  # pylint: disable=invalid-name

NoneType = type(None)
T = TypeVar("T")


class Enum(enum.Enum):
    """
    A subclass of the builtin :class:`enum.Enum` class, but uses the lower-cased names
    as enum values when used with ``auto()``.  For example::

        from argtyped import Enum, auto

        class MyEnum(Enum):
            OPTION_A = auto()
            OPTION_B = auto()

    is equivalent to::

        from enum import Enum

        class MyEnum(Enum):
            OPTION_A = "option_a"
            OPTION_B = "option_b"
    """

    @staticmethod
    def _generate_next_value_(
        name: str, start: int, count: int, last_values: List[str]
    ) -> str:
        return name.lower()

    def __eq__(self, other: object) -> bool:
        return self.value == other or super().__eq__(other)


# Switch is a type that's different but equivalent to `bool`.
# It is defined as the `Union` of `bool` and a dummy type, because:
# 1. `bool` cannot be sub-typed.
#    >> Switch = type('Switch', (bool,), {})
# 2. `Union` with a single (possibly duplicated) type is flattened into that type.
#    >> Switch = Union[bool]
# 3. `NewType` forbids implicit casts from `bool`.
#    >> Switch = NewType('Switch', bool)
__dummy_type__ = type(  # pylint: disable=invalid-name
    "__dummy_type__", (), {}  # names must match for pickle to work
)
Switch = Union[bool, __dummy_type__]  # type: ignore[valid-type]


def is_choices(typ: type) -> bool:
    r"""
    Check whether a type is a choices type (:class:`Choices` or :class:`Literal`).
    This cannot be checked using traditional methods,  since :class:`Choices` is a
    metaclass.
    """
    return typing_inspect.is_literal_type(typ)


def unwrap_choices(typ: type) -> Tuple[str, ...]:
    r"""
    Return the string literals associated with the choices type. Literal type in
    Python 3.7+ stores the literals in ``typ.__args__``, but in Python 3.6- it's in
    ``typ.__values__``.
    """
    return typing_inspect.get_args(typ, evaluate=True)


def is_enum(typ: Any) -> bool:
    r"""
    Check whether a type is an Enum type. Since we're using ``issubclass``, we need to
    check whether :arg:`typ` is a type first.
    """
    return isinstance(typ, type) and issubclass(typ, enum.Enum)


def is_optional(typ: type) -> bool:
    r"""
    Check whether a type is `Optional[T]`. `Optional` is internally implemented as
    `Union` with `type(None)`.
    """
    return typing_inspect.is_optional_type(typ)


def is_list(typ: type) -> bool:
    r"""Check whether a type if `List[T]`."""
    # Note: The origin is `List` in Python 3.6, and `list` in Python 3.7+.
    return typing_inspect.get_origin(typ) in (list, List)


def unwrap_optional(typ: Type[Optional[T]]) -> Type[T]:
    r"""Return the inner type inside an `Optional[T]` type."""
    # Note: In Python 3.6, `get_args` returns a tuple if `evaluate` is not set to True,
    # due to it having a different internal representation.  For compatibility, we need
    # to always set `evaluate` to True.
    remain_types = [
        t for t in typing_inspect.get_args(typ, evaluate=True) if t is not NoneType
    ]
    if len(remain_types) >= 2:
        if set(remain_types) == set(typing_inspect.get_args(Switch, evaluate=True)):
            return Switch  # type: ignore[return-value]
        raise TypeError(f"Invalid type {typ}: 'Union' types are not supported")
    return remain_types[0]


def unwrap_list(typ: Type[List[T]]) -> Type[T]:
    r"""Return the inner type inside an `List[T]` type."""
    return typing_inspect.get_args(typ, evaluate=True)[0]
