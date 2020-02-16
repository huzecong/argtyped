# `argtyped`: Command Line Argument Parser, with Types

[![Build Status](https://github.com/huzecong/argtyped/workflows/Build/badge.svg)](https://github.com/huzecong/argtyped/actions?query=workflow%3ABuild+branch%3Amaster)
[![CodeCov](https://codecov.io/gh/huzecong/argtyped/branch/master/graph/badge.svg?token=ELHfYJ2Ydq)](https://codecov.io/gh/huzecong/argtyped)
[![PyPI](https://badge.fury.io/py/argtyped.svg)](https://pypi.org/project/argtyped/)
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
parser.add_argument("--logging-level", choices=list(LoggingLevels), type=LoggingLevels, default="info")

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
Then the parsed arguments will be equivalent to the following structured returned by `argparse`:
```python
argparse.Namespace(
    model_name="LSTM", hidden_size=512, activation="sigmoid", logging_level="debug",
    use_dropout=False, dropout_prob=None)
```

Arguments can also be pretty-printed with `print(args.to_string())`, which gives:
```
<class '__main__.MyArguments'>
╔═════════════════╤══════════════════════════════════╗
║ Arguments       │ Values                           ║
╠═════════════════╪══════════════════════════════════╣
║ model_name      │ 'LSTM'                           ║
║ hidden_size     │ 512                              ║
║ activation      │ 'sigmoid'                        ║
║ logging_level   │ <MyLoggingLevels.Debug: 'debug'> ║
║ use_dropout     │ False                            ║
║ dropout_prob    │ None                             ║
║ label_smoothing │ 0.1                              ║
║ some_true_arg   │ True                             ║
║ some_false_arg  │ False                            ║
╚═════════════════╧══════════════════════════════════╝
```

## Reference

### The `argtyped.Arguments` Class

The `argtyped.Arguments` class is main class of the package, from which you should derive your custom class that holds
arguments. Each argument takes the form of a class attribute, with its type annotation and an optional default value.

When an instance of your custom class is initialized, the command line arguments are parsed from `sys.argv` into values
with your annotated types. You can also provide the list of strings to parse by passing them as the parameter.

The parsed arguments are stored in an object of your custom type. This gives you arguments that can be auto-completed
by the IDE, and type-checked by a static type checker like [`mypy`](http://mypy-lang.org/).

The following example illustrates the keypoints:
```python
class MyArgs(argtyped.Arguments):
    # name: type [= default_val]
    value: int = 0

args = MyArgs()                    # equivalent to `parser.parse_args()`
args = MyArgs(["--value", "123"])  # equivalent to `parser.parse_args(["--value", "123"])
assert isinstance(args, MyArgs)
```

### Argument Types

To summarize, whatever works for `argparse` works here. The following types are supported:

- Built-in types such as `int`, `float`, `str`.
- `bool` type. Accepted values (case-insensitive) for `True` are: `y`, `yes`, `true`, `ok`; accepted values for `False`
  are: `n`, `no`, `false`.
- Choice types `Choices[...]`. A choice argument is essentially an `str` argument with limited choice of values. The
  ellipses can be filled with a tuple of `str`s, or an expression that evaluates to a list of `str`s:
  ```python
  from argtyped import Arguments, Choices
  from typing import List

  def logging_levels() -> List[str]:
      return ["debug", "info", "warning", "error"]

  class MyArgs(Arguments):
      foo: Choices["debug", "info", "warning", "error"]  # 4 choices
      bar: Choices[logging_levels()]                     # the same 4 choices

  # argv: ["--foo=debug", "--bar=info"] => foo="debug", bar="info"
  ```
  This is equivalent to the `choices` keyword in `argparse.add_argument`.
- Enum types derived from `enum.Enum`. It is recommended to use `argtyped.Enum` which uses the instance names as values:
  ```python
  from argtyped import Enum

  class MyEnum(Enum):
      Debug = auto()    # "debug"
      Info = auto()     # "info"
      Warning = auto()  # "warning"
  ```
- Switch types `Switch`. `Switch` arguments are like `bool` arguments, but they don't take values. Instead, a switch
  argument `switch` requires `--switch` to enable and `--no-switch` to disable:
  ```python
  from argtyped import Arguments, Switch

  class MyArgs(Arguments):
      switch: Switch = True
      bool_arg: bool = False

  # argv: []                                 => flag=True, bool_arg=False
  # argv: ["--switch", "--bool-arg=false"]   => flag=True, bool_arg=False
  # argv: ["--no-switch", "--bool-arg=true"] => flag=False, bool_arg=True
  # argv: ["--switch=false"]                 => WRONG
  # argv: ["--no-bool-arg"]                  => WRONG
  ```
- Optional types `Optional[T]`, where `T` is any supported type (except choices or switch). An optional argument will be
  filled with `None` if no value is provided. It could also be explicitly set to `None` by using `none` as value in the
  command line:
  ```python
  from argtyped import Arguments
  from typing import Optional

  class MyArgs(Arguments):
      opt_arg: Optional[int]

  # argv: []                 => opt_arg=None
  # argv: ["--opt-arg=1"]    => opt_arg=1
  # argv: ["--opt-arg=none"] => opt_arg=None
  ```
- Any other type that takes a single `str` as `__init__` parameters. It is also theoretically possible to use a function
  that takes an `str` as input, but it's not recommended as it's not type-safe.


## Caveats

- Advanced `argparse` features such as subparsers, groups, argument lists, and custom actions are not supported.
- Using switch arguments may result in name clashes: if a switch argument has name `arg`, there can be no argument with
  the name `no_arg`.
- `Optional` cannot be used with `Choices`. You can add `"none"` as a valid choice to mimic a similar behavior.

## Under the Hood

This is what happens under the hood:
1. When an instance of `argtyped.Arguments` (or any subclass) is initialized, type annotations and class-level
   attributes (i.e., the default values) are collected to form argument declarations.
2. After verifying the validity of declared arguments, an instance of `argparse.ArgumentParser` is created and arguments
   are registered with the parser.
3. The parser's `parse_args` method is invoked with either `sys.argv` or strings provided as parameters, returning
   parsed arguments.
4. The parsed arguments are assigned to `self` (the instance of `Arguments` subclass being initialized).
