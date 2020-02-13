# `argtyped`: Command Line Argument Parser, with Types

[![Build Status](https://github.com/huzecong/argtyped/workflows/Build/badge.svg)](https://github.com/huzecong/argtyped/actions?query=workflow%3ABuild+branch%3Amaster)
[![CodeCov](https://codecov.io/gh/huzecong/argtyped/branch/master/graph/badge.svg?token=ELHfYJ2Ydq)](https://codecov.io/gh/huzecong/argtyped)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/huzecong/argtyped/blob/master/LICENSE)

`argtyped` is an command line argument parser with that relies on type annotations. It is built on
[`argparse`](https://docs.python.org/3/library/argparse.html), the command line argument parser library built into
Python. Compared with `argparse`, this library gives you:

- More concise and intuitive syntax, less boilerplate code.
- Type checking and IDE auto-completion for command line arguments.
- A drop-in replacement for `argparse` in most cases.


## Installation

Install stable release from [PyPI](https://pypi.org/project/argtyped/):
```bash
pip install argtyped
```

Or, install the latest commit from GitHub:
```bash
pip install -e git+https://github.com/huzecong/argtyped.git
```

## Usage

With `argtyped`, you can define command line arguments in a syntax similar to
[`typing.NamedTuple`](https://docs.python.org/3/library/typing.html#typing.NamedTuple). The syntax is intuitive and can
be illustrated with an example:
```python
from typing import Optional

from argtyped import Arguments, Choices, Switch
from argtyped import Enum, auto

class LoggingLevels(Enum):
    Debug = auto()
    Info = auto()
    Warning = auto()
    Error = auto()
    Critical = auto()

class MyArguments(Arguments):
    model_name: str         # required argument of `str` type
    hidden_size: int = 512  # `int` argument with default value of 512
    
    activation: Choices['relu', 'tanh', 'sigmoid'] = 'relu'  # argument with limited choices
    logging_level: LoggingLevels = LoggingLevels.Info        # using `Enum` class as choices

    use_dropout: Switch = True  # switch argument, enable with "--use-dropout" and disable with "--no-use-dropout"
    dropout_prob: Optional[float] = 0.5  # optional argument, "--dropout-prob=none" parses into `None`

args = Arguments()
```

This is equivalent to the following code with Python built-in `argparse`:
```python
import argparse
from enum import Enum

class LoggingLevels(Enum):
    Debug = "debug"
    Info = "info"
    Warning = "warning"
    Error = "error"
    Critical = "critical"

parser = argparse.ArgumentParser()

parser.add_argument("--model-name", type=str, required=True)
parser.add_argument("--hidden-size", type=int, default=512)

parser.add_argument("--activation", choices=["relu", "tanh", "sigmoid"], default="relu")
parser.add_argument("--logging-level", choices=[item.value for item in LoggingLevels], default="info")

parser.add_argument("--use-dropout", action="store_true", dest="use_dropout", default=True)
parser.add_argument("--no-use-dropout", action="store_false", dest="use_dropout")
parser.add_argument("--dropout-prob", type=lambda s: None if s.lower() == 'none' else float(s), default=0.5)

args = parser.parse_args()
```

Save the code into a file named `main.py`. Suppose the following arguments are provided:
```bash
python main.py \
    --model-name LSTM \
    --activation sigmoid \
    --logging-level debug \
    --no-use-dropout \
    --dropout-prob none
```
Then the parsed arguments will be:
```python
argparse.Namespace(
    model_name="LSTM", hidden_size=512, activation="sigmoid", logging_level="debug",
    use_dropout=False, dropout_prob=None)
```

## Reference

## Caveats

**Note:** Advanced features such as subparsers, groups, argument lists, custom actions are not supported.

## Under the Hood
