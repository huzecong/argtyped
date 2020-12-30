from enum import Enum as BuiltinEnum, auto
from typing import Optional, Union

from typing_extensions import Literal

from argtyped import Choices, Enum as ArgtypedEnum
from argtyped.custom_types import (
    is_choices,
    is_enum,
    is_optional,
    unwrap_choices,
    unwrap_optional,
)


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
    assert not is_enum(Choices)
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
    assert unwrap_optional(Optional[literal]) is literal
    assert unwrap_optional(Union[int, type(None)]) is int


def test_is_choices():
    assert is_choices(Choices["a"])
    assert is_choices(Choices["a", "b", "c"])
    assert is_choices(Choices[["a", "b"] + ["c"]])
    assert is_choices(Literal["a"])
    assert is_choices(Literal["a", "b", "c"])
    assert not is_choices(Union[int, float, "c"])
    assert not is_choices(Optional["a"])


def test_unwrap_choices():
    assert unwrap_choices(Choices["a"]) == ("a",)
    assert unwrap_choices(Choices["a", "b", "c"]) == ("a", "b", "c")
    assert unwrap_choices(Choices[["a", "b"] + ["c"]]) == ("a", "b", "c")
    assert unwrap_choices(Literal["a", "b", "c"]) == ("a", "b", "c")
