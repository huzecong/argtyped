from typing import List, Optional, Union

import pytest
from typing_extensions import Literal

from argtyped.arguments import Arguments
from argtyped.custom_types import Enum, Switch, auto


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
    with pytest.raises(TypeError, match=r"must have a boolean default value"):

        class Args(Arguments):
            a: Switch = True
            b: Switch  # wrong
            c: Switch = 0  # wrong


def test_invalid_choice():
    with pytest.raises(TypeError, match=r"must be string"):

        class Args1(Arguments):
            a: Literal["1", 2]


def test_invalid_list():
    with pytest.raises(TypeError, match="must be of list type"):

        class Args1(Arguments):
            a: List[Optional[int]] = None


def test_invalid_nesting():
    with pytest.raises(TypeError, match="'List' cannot be nested inside 'List'"):

        class Args1(Arguments):
            a: List[List[int]]

    with pytest.raises(TypeError, match="'List' cannot be nested inside 'Optional'"):

        class Args2(Arguments):
            a: Optional[List[int]]

    with pytest.raises(TypeError, match="'Switch' cannot be nested inside 'List'"):

        class Args3(Arguments):
            a: List[Switch]

    with pytest.raises(TypeError, match="'Switch' cannot be nested inside 'Optional'"):

        class Args4(Arguments):
            a: Optional[Switch]

    with pytest.raises(TypeError, match="cannot be nested"):

        class Args5(Arguments):
            a: List[Optional[List[int]]]

    with pytest.raises(TypeError, match=r"'Union'.*not supported"):

        class Args6(Arguments):
            a: Optional[Union[int, float]]


def test_invalid_bool(catch_parse_error):
    class Args(Arguments):
        a: bool

    _ = Args(["--a", "True"])
    _ = Args(["--a", "Y"])
    with catch_parse_error(r"Invalid value .* for bool"):
        _ = Args(["--a=nah"])


def test_invalid_type():
    with pytest.raises(TypeError, match="invalid type"):

        class Args1(Arguments):
            a: 5 = 0

    with pytest.raises(TypeError, match=r"forward reference.*not.*supported"):

        class Args2(Arguments):
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
