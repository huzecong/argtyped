# `argtyped`: `argparse` with Types

![build-badge]()

`argtyped` adds type annotations to [`argparse`](https://docs.python.org/3/library/argparse.html), the command line
argument parser library built into Python. Compared to `argparse`, this library gives you:

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
    Debug = auto()  # equivalent to `Debug = "debug"`
    Info = auto()
    Warning = auto()
    Error = auto()
    Critical = auto()

class MyArguments(Arguments):
    # Define a required argument of type `str`.
    model_name: str
    # Define an `int` argument with default value of 512.
    hidden_size: int = 512
    # Define an argument with a limited number of choices, equivalent to the "choice" keyword in `argparse`.
    activation: Choices['relu', 'tanh', 'sigmoid'] = 'relu'
    # Define an argument of an `Enum` type. The command line string "critical" will be parsed into
    # `LoggingLevels.Critical`.
    logging_level: LoggingLevels = LoggingLevels.Info
    # Define a "switch" argument. In this case, the flag "--use-dropout" sets the value to `True`, and the flag
    # "--no-use-dropout" sets it to `False`. 
    use_dropout: Switch = True
    # Define an optional argument. The command line string "none" will be parsed into `None`. 
    dropout_prob: Optional[float] = 0.5

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
parser.add_argument("--logging-level", choices=[str(item) for item in LoggingLevels], default="info")
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
