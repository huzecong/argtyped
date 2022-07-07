from typing import List, Optional

import attr

from argtyped.attrs_arguments import AttrsArguments, positional_arg


def test_attrs_attributes():
    def _convert_d(x: str) -> Optional[str]:
        return None if x == "NOTHING" else x

    @attr.s(auto_attribs=True, kw_only=True)
    class AttrsArgs(AttrsArguments):
        a: int
        b: Optional[float] = None
        c: List[str] = attr.ib()
        d: Optional[str] = attr.ib(converter=_convert_d)
        e: int = 3

    args = AttrsArgs.parse_args("--a 1 --c x y z --d=none".split())
    assert args == AttrsArgs(a=1, c=["x", "y", "z"], d="none")


def test_attrs_nullable_required(catch_parse_error):
    @attr.s
    class TestArgs(AttrsArguments):
        a: Optional[int] = attr.ib()

    with catch_parse_error():
        _ = TestArgs.parse_args([])
    args = TestArgs.parse_args(["--a=none"])
    assert args == TestArgs(a=None)


def test_attrs_positional_arguments():
    @attr.s
    class PositionalArgs(AttrsArguments):
        a: str = attr.ib(metadata={"positional": True})
        b: int = positional_arg(default=2)
        c: float = positional_arg(default=1.2)

    args = PositionalArgs.parse_args("x 1 0.1".split())
    assert args == PositionalArgs(a="x", b=1, c=0.1)

    args = PositionalArgs.parse_args("x 1".split())
    assert args == PositionalArgs(a="x", b=1)

    args = PositionalArgs.parse_args(["x"])
    assert args == PositionalArgs(a="x")


def test_attrs_custom_argparse_options(catch_parse_error):
    def _convert_c(xss: List[List[str]]) -> List[int]:
        # In Python 3.8+ we'd just set `action="extend"`.
        return [int(x) for xs in xss for x in xs]

    @attr.s
    class CustomOptionArgs(AttrsArguments):
        a: List[int] = positional_arg(metadata={"nargs": "+"})
        b: List[int] = attr.ib()
        c: List[int] = attr.ib(metadata={"action": "append"}, converter=_convert_c)

    with catch_parse_error():
        _ = CustomOptionArgs.parse_args("--b --c".split())
    args = CustomOptionArgs.parse_args("1 2 --b --c".split())
    assert attr.asdict(args) == {"a": [1, 2], "b": [], "c": []}

    with catch_parse_error():
        _ = CustomOptionArgs.parse_args("1 --b 1 2 --b 3 4".split())
    args = CustomOptionArgs.parse_args("1 --c 1 2 --c 3 --b 1 2 --c 4 5".split())
    assert attr.asdict(args) == {"a": [1], "b": [1, 2], "c": [1, 2, 3, 4, 5]}
