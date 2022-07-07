from typing import Optional

from typing_extensions import Literal

from argtyped.arguments import ArgumentKind, Arguments, ArgumentSpec, argument_specs
from argtyped.custom_types import Enum, Switch, auto


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
    b: Literal["a", "b", "c"] = "b"  # type: ignore
    c: MyEnum  # type: ignore  # override base argument w/ default


class FinalArgs(DerivedArgs, underscore=True):
    final_arg: str


def test_underscore_inheritance():
    underscore_args = {"underscore_arg", "underscore_switch", "final_arg"}
    for name, spec in argument_specs(FinalArgs).items():
        assert spec.underscore == (name in underscore_args)


def test_argument_specs():
    base_specs = {
        "a": ArgumentSpec(
            kind=ArgumentKind.NORMAL, nullable=False, required=True, type=int
        ),
        "b": ArgumentSpec(
            kind=ArgumentKind.NORMAL,
            nullable=True,
            required=False,
            type=bool,
            default=None,
        ),
        "c": ArgumentSpec(
            kind=ArgumentKind.NORMAL,
            nullable=False,
            required=False,
            type=str,
            default="abc",
        ),
        "d": ArgumentSpec(
            kind=ArgumentKind.SWITCH,
            nullable=False,
            required=False,
            type=bool,
            default=False,
        ),
    }
    underscore_specs = {
        "underscore_arg": ArgumentSpec(
            kind=ArgumentKind.NORMAL,
            nullable=False,
            required=True,
            type=int,
            underscore=True,
        ),
        "underscore_switch": ArgumentSpec(
            kind=ArgumentKind.SWITCH,
            nullable=False,
            required=False,
            type=bool,
            default=True,
            underscore=True,
        ),
    }
    derived_specs = {
        **{name: specs._replace(inherited=True) for name, specs in base_specs.items()},
        **{
            name: specs._replace(inherited=True)
            for name, specs in underscore_specs.items()
        },
        "b": ArgumentSpec(
            kind=ArgumentKind.NORMAL,
            nullable=False,
            required=False,
            type=str,
            choices=("a", "b", "c"),
            default="b",
        ),
        "c": ArgumentSpec(
            kind=ArgumentKind.NORMAL,
            nullable=False,
            required=True,
            type=MyEnum,
            choices=(MyEnum.A, MyEnum.B),
        ),
        "e": ArgumentSpec(
            kind=ArgumentKind.NORMAL, nullable=False, required=True, type=float
        ),
    }
    assert dict(argument_specs(BaseArgs)) == base_specs
    assert dict(argument_specs(DerivedArgs)) == derived_specs

    # Test parsing
    _ = DerivedArgs(
        "--a 1 --e 1.0 --c a --underscore_arg 1 --no_underscore_switch".split()
    )


def test_correct_resolution_order():
    """
    Test that attributes are collected using the correct MRO, instead of simply looping
    over the base classes.  The wrong base class approach will incorrectly set `y`'s
    type to `int`.
    """

    class A(Arguments):
        x: int
        y: int

    class B(A):
        x: float  # type: ignore

    class C(A):
        y: float  # type: ignore

    class D(B, C):
        pass

    specs = argument_specs(D)
    assert set(specs.keys()) == {"x", "y"}
    assert specs["x"].type == float
    assert specs["y"].type == float
