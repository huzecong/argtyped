from typing import Optional

from typing_extensions import Literal

from argtyped import *


class MyEnum(Enum):
    A = auto()
    B = auto()


class BaseArgs(Arguments):
    a: int
    b: Optional[bool]
    c: str = "abc"
    d: Switch = False


class DerivedArgs(BaseArgs):
    e: float
    b: Literal["a", "b", "c"] = "b"
    c: MyEnum


def test_argument_specs():
    base_specs = {
        "a": ArgumentSpec(type="normal", nullable=False, required=True, value_type=int),
        "b": ArgumentSpec(
            type="normal", nullable=True, required=False, value_type=bool, default=None
        ),
        "c": ArgumentSpec(
            type="normal", nullable=False, required=False, value_type=str, default="abc"
        ),
        "d": ArgumentSpec(
            type="switch",
            nullable=False,
            required=False,
            value_type=bool,
            default=False,
        ),
    }
    derived_specs = {
        **base_specs,
        "b": ArgumentSpec(
            type="normal",
            nullable=False,
            required=False,
            value_type=str,
            choices=("a", "b", "c"),
            default="b",
        ),
        "c": ArgumentSpec(
            type="normal",
            nullable=False,
            required=True,
            value_type=MyEnum,
            choices=("A", "B"),
        ),
        "e": ArgumentSpec(
            type="normal", nullable=False, required=True, value_type=float
        ),
    }
    assert dict(argument_specs(BaseArgs)) == base_specs
    assert dict(argument_specs(DerivedArgs)) == derived_specs
