from typing import Optional

import argparse
import pytest

from argtyped import *


def test_no_type_annotation():
    class Args(Arguments):
        a = 1  # wrong
        b = 2  # wrong
        c: int = 3

    with pytest.raises(ValueError, match=r"does not have type annotation"):
        _ = Args()


def test_non_nullable():
    class Args(Arguments):
        a: Optional[int] = None
        b: Optional[int]
        c: int = None  # wrong

    with pytest.raises(ValueError, match=r"not nullable"):
        _ = Args()


def test_switch_not_bool():
    class Args(Arguments):
        a: Switch = True
        b: Switch  # wrong
        c: Switch = 0  # wrong

    with pytest.raises(ValueError, match=r"[Ss]witch argument .* default value of type bool"):
        _ = Args()


def test_invalid_choice():
    with pytest.raises(TypeError, match=r"must contain at least one"):
        class _Args(Arguments):
            a: Choices[()]

    class Args(Arguments):
        a: Choices['a', 'b'] = "c"  # wrong

    with pytest.raises(ValueError, match=r"[Ii]nvalid default value for choice"):
        _ = Args()


def test_invalid_bool():
    class Args(Arguments):
        a: bool

    _ = Args(["--a", "True"])
    _ = Args(["--a", "Y"])
    with pytest.raises(argparse.ArgumentError, match=r"[Ii]nvalid .* bool"):
        try:
            _ = Args(["--a=nah"])
        except SystemExit:
            pass
