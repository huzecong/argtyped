import argparse
import enum
import pickle
from typing import Optional

import pytest
from typing_extensions import Literal

from argtyped import *
from argtyped.arguments import _TYPE_CONVERSION_FN


class MyLoggingLevels(Enum):
    Debug = auto()
    Info = auto()
    Warning = auto()
    Error = auto()
    Critical = auto()


class LoggingLevels(enum.Enum):
    Debug = "debug"
    Info = "info"
    Warning = "warning"
    Error = "error"
    Critical = "critical"

    def __eq__(self, other):
        if isinstance(other, MyLoggingLevels):
            return self.value == other.value
        return super().__eq__(other)


class MyArguments(Arguments):
    model_name: str
    hidden_size: int = 512
    activation: Choices["relu", "tanh", "sigmoid"] = "relu"
    activation2: Literal["relu", "tanh", "sigmoid"] = "tanh"
    logging_level: MyLoggingLevels = MyLoggingLevels.Info
    use_dropout: Switch = True
    dropout_prob: Optional[float] = 0.5
    label_smoothing: Optional[float]
    some_true_arg: bool
    some_false_arg: bool


CMD = r"""
    --model-name LSTM
    --activation sigmoid
    --activation2=sigmoid
    --logging-level=debug
    --no-use-dropout
    --dropout-prob none
    --label-smoothing 0.1
    --some-true-arg=yes
    --some-false-arg n
    """.split()

RESULT = dict(
    model_name="LSTM",
    hidden_size=512,
    activation="sigmoid",
    activation2="sigmoid",
    logging_level=MyLoggingLevels.Debug,
    use_dropout=False,
    dropout_prob=None,
    label_smoothing=0.1,
    some_true_arg=True,
    some_false_arg=False,
)


def test_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument(
        "--activation", choices=["relu", "tanh", "sigmoid"], default="relu"
    )
    parser.add_argument(
        "--activation2", choices=["relu", "tanh", "sigmoid"], default="tanh"
    )
    parser.add_argument(
        "--logging-level",
        choices=list(LoggingLevels),
        type=LoggingLevels,
        default="info",
    )
    parser.add_argument(
        "--use-dropout", action="store_true", dest="use_dropout", default=True
    )
    parser.add_argument("--no-use-dropout", action="store_false", dest="use_dropout")
    parser.add_argument(
        "--dropout-prob",
        type=lambda s: None if s.lower() == "none" else float(s),
        default=0.5,
    )
    parser.add_argument(
        "--label-smoothing",
        type=lambda s: None if s.lower() == "none" else float(s),
        default=None,
    )
    parser.add_argument(
        "--some-true-arg", type=_TYPE_CONVERSION_FN[bool], required=True
    )
    parser.add_argument(
        "--some-false-arg", type=_TYPE_CONVERSION_FN[bool], required=True
    )

    namespace = parser.parse_args(CMD)
    assert isinstance(namespace, argparse.Namespace)
    for key in RESULT:
        assert RESULT[key] == getattr(namespace, key)
    args = MyArguments(CMD)
    assert isinstance(args, MyArguments)
    for key in RESULT:
        assert RESULT[key] == getattr(args, key)

    assert args.to_dict() == RESULT


def test_print():
    args = MyArguments(CMD)
    width = 50

    output = args.to_string(width)
    for line in output.strip().split("\n")[1:]:  # first line is class type
        assert len(line) == width
    for key in args._annotations:
        assert key in output

    output = args.to_string(max_width=width)
    for line in output.strip().split("\n")[1:]:
        assert len(line) <= width

    with pytest.raises(ValueError, match=r"must be None"):
        args.to_string(width, width)
    with pytest.raises(ValueError, match=r"cannot be drawn"):
        invalid_width = max(map(len, args._annotations)) + 7 + 6 - 1
        args.to_string(invalid_width)


def test_pickle():
    args = MyArguments(CMD)
    args_restored = pickle.loads(pickle.dumps(args))
    for key in RESULT:
        assert RESULT[key] == getattr(args_restored, key)
