import sys
from enum import Enum as BuiltinEnum
from enum import auto
from typing import List, Optional, Tuple, TypeVar, Union

from typing_extensions import Literal

from argtyped import Enum as ArgtypedEnum
from argtyped.custom_types import (
    is_choices,
    is_enum,
    is_list,
    is_optional,
    unwrap_choices,
    unwrap_list,
    unwrap_optional,
)

T = TypeVar("T")


def test_is_enum():
    class A(ArgtypedEnum):
        foo = auto()
        bar = auto()

    class B(BuiltinEnum):
        foo = auto()
        bar = auto()

    assert is_enum(A)
    assert is_enum(B)
    assert not is_enum(1234)
    assert not is_enum(Optional[A])


def test_is_optional():
    assert is_optional(Optional[int])
    assert is_optional(Union[Optional[int], Optional[float]])
    assert is_optional(Optional[Union[int, float]])
    assert is_optional(Union[int, None])
    assert is_optional(Union[int, type(None)])
    assert is_optional(Optional["int"])
    assert not is_optional(Optional)
    assert not is_optional(Union[int, float])


def test_unwrap_optional():
    assert unwrap_optional(Optional[int]) is int
    assert unwrap_optional(Optional["int"]).__forward_arg__ == "int"
    literal = Literal["a", "b", "c"]
    assert unwrap_choices(unwrap_optional(Optional[literal])) == unwrap_choices(literal)
    assert unwrap_optional(Union[int, type(None)]) is int


def test_is_choices():
    assert is_choices(Literal["a"])
    assert is_choices(Literal["a", "b", "c"])
    assert not is_choices(Union[int, float, "c"])
    assert not is_choices(Optional["a"])
    assert not is_choices(Optional[Literal["a"]])


def test_unwrap_choices():
    assert unwrap_choices(Literal["a"]) == ("a",)
    assert unwrap_choices(Literal["a", "b", "c"]) == ("a", "b", "c")


def test_is_list():
    assert is_list(List[int])
    assert is_list(List[T][int])
    assert is_list(List[Optional[int]])
    assert is_list(List[List[int]])
    assert is_list(List[Union[int, float]])
    assert not is_list(Optional[List[Optional[int]]])
    assert not is_list(Tuple[int, int])
    if sys.version_info >= (3, 9):
        # PEP-585 was implemented in Python 3.9, which allows using standard types as
        # typing containers.
        assert is_list(list[int])
        assert is_list(list[T][int])
        assert not is_list(Optional[list[int]])
        assert not is_list(dict[str, list[int]])
        assert not is_list(tuple[int, int])


def test_unwrap_list():
    assert unwrap_list(List[int]) == int
    assert unwrap_list(List[T][int]) == int
    assert unwrap_optional(unwrap_list(List[Optional[int]])) == int
    if sys.version_info >= (3, 9):
        assert unwrap_list(list[int]) == int
        assert unwrap_list(list[T][int]) == int
        assert unwrap_optional(unwrap_list(list[Optional[int]])) == int
