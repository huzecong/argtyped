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


class UnderscoreArgs(Arguments, underscore=True):
    underscore_arg: int
    underscore_switch: Switch = True


class DerivedArgs(BaseArgs, UnderscoreArgs):
    e: float
    b: Literal["a", "b", "c"] = "b"
    c: MyEnum  # override base argument w/ default


class FinalArgs(DerivedArgs, underscore=True):
    final_arg: str


def test_underscore_inheritance():
    underscore_args = {"underscore_arg", "underscore_switch", "final_arg"}
    for name, spec in argument_specs(FinalArgs).items():
        assert spec.underscore == (name in underscore_args)


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
    underscore_specs = {
        "underscore_arg": ArgumentSpec(
            type="normal",
            nullable=False,
            required=True,
            value_type=int,
            underscore=True,
        ),
        "underscore_switch": ArgumentSpec(
            type="switch",
            nullable=False,
            required=False,
            value_type=bool,
            default=True,
            underscore=True,
        ),
    }
    derived_specs = {
        **base_specs,
        **underscore_specs,
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
            choices=(MyEnum.A, MyEnum.B),
        ),
        "e": ArgumentSpec(
            type="normal", nullable=False, required=True, value_type=float
        ),
    }
    assert dict(argument_specs(BaseArgs)) == base_specs
    assert dict(argument_specs(DerivedArgs)) == derived_specs

    # Test parsing
    _ = DerivedArgs(
        "--a 1 --e 1.0 --c a --underscore_arg 1 --no_underscore_switch".split()
    )
