import re
from typing import Optional

import pytest
from typing_extensions import Literal

from argtyped import *


def test_no_type_annotation():
    with pytest.raises(TypeError, match=r"does not have type annotation"):

        class Args(Arguments):
            a = 1  # wrong
            b = 2  # wrong
            c: int = 3


def test_non_nullable():
    with pytest.raises(TypeError, match=r"not nullable"):

        class Args(Arguments):
            a: Optional[int] = None
            b: Optional[int]
            c: int = None  # wrong


def test_switch_not_bool():
    with pytest.raises(
        TypeError, match=r"Switch argument .* default value of type bool"
    ):

        class Args(Arguments):
            a: Switch = True
            b: Switch  # wrong
            c: Switch = 0  # wrong


def test_invalid_choice():
    with pytest.raises(TypeError, match=r"must contain at least one"):

        class Args1(Arguments):
            a: Choices[()]

    with pytest.raises(TypeError, match=r"invalid default value"):

        class Args2(Arguments):
            a: Choices["a"] = "b"  # wrong

    with pytest.raises(TypeError, match=r"must be string"):

        class Args3(Arguments):
            a: Choices["1", 2]

    with pytest.raises(TypeError, match=r"must be string"):

        class Args4(Arguments):
            a: Literal["1", 2]


def test_invalid_bool(catch_parse_error):
    class Args(Arguments):
        a: bool

    _ = Args(["--a", "True"])
    _ = Args(["--a", "Y"])
    with catch_parse_error(r"Invalid value .* for bool"):
        _ = Args(["--a=nah"])


def test_invalid_type():
    with pytest.raises(TypeError, match="invalid type"):

        class Args(Arguments):
            a: 5 = 0

    with pytest.raises(TypeError, match="invalid type"):

        class Args(Arguments):
            b: "str" = 1


def test_invalid_enum():
    class MyEnum(Enum):
        a = auto()
        b = auto()

    with pytest.raises(TypeError, match="invalid default value"):

        class Args1(Arguments):
            enum: MyEnum = "c"

        class Args(Arguments):
            enum: MyEnum = "b"  # must be enum value, not string
