# `argtyped`: Command Line Argument Parser, with Types

[![Build Status](https://github.com/huzecong/argtyped/workflows/Build/badge.svg)](https://github.com/huzecong/argtyped/actions?query=workflow%3ABuild+branch%3Amaster)
[![CodeCov](https://codecov.io/gh/huzecong/argtyped/branch/master/graph/badge.svg?token=ELHfYJ2Ydq)](https://codecov.io/gh/huzecong/argtyped)
[![PyPI](https://img.shields.io/pypi/v/argtyped.svg)](https://pypi.org/project/argtyped/)
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
pip install git+https://github.com/huzecong/argtyped.git
```

## Usage

With `argtyped`, you can define command line arguments in a syntax similar to
[`typing.NamedTuple`](https://docs.python.org/3/library/typing.html#typing.NamedTuple). The syntax is intuitive and can
be illustrated with an example:
```python
from typing import Optional
from typing_extensions import Literal  # or directly import from `typing` in Python 3.8+

from argtyped import Arguments, Switch
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

    activation: Literal['relu', 'tanh', 'sigmoid'] = 'relu'  # argument with limited choices
    logging_level: LoggingLevels = LoggingLevels.Info        # using `Enum` class as choices

    use_dropout: Switch = True  # switch argument, enable with "--use-dropout" and disable with "--no-use-dropout"
    dropout_prob: Optional[float] = 0.5  # optional argument, "--dropout-prob=none" parses into `None`

args = MyArguments()
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
Then the parsed arguments will be equivalent to the following structure returned by `argparse`:
```python
argparse.Namespace(
    model_name="LSTM", hidden_size=512, activation="sigmoid", logging_level="debug",
    use_dropout=False, dropout_prob=None)
```

Arguments can also be pretty-printed:
```
>>> print(args)
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
It is recommended though to use the `args.to_string()` method, which gives you control of the table width.

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

#### `Arguments.to_dict(self)`

Convert the set of arguments to a dictionary (`OrderedDict`).

#### `Arguments.to_string(self, width: Optional[int] = None, max_width: Optional[int] = None)`

Represent the arguments as a table.
- `width`: Width of the printed table. Defaults to `None`, which fits the table to its contents. An exception is raised
  when the table cannot be drawn with the given width.
- `max_width`: Maximum width of the printed table. Defaults to `None`, meaning no limits. Must be `None` if `width` is
  not `None`.

#### `argtyped.argument_specs`

Return a dictionary mapping argument names to their specifications, represented as the `argtyped.ArgumentSpec` type.
This is useful for programmatically accessing the list of arguments.

### Argument Types

To summarize, whatever works for `argparse` works here. The following types are supported:

- **Built-in types** such as `int`, `float`, `str`.
- **Boolean type** `bool`. Accepted values (case-insensitive) for `True` are: `y`, `yes`, `true`, `ok`; accepted values
  for `False` are: `n`, `no`, `false`.
- **Choice types** `Literal[...]`. A choice argument is essentially an `str` argument with limited
  choice of values. The ellipses can be filled with a tuple of `str`s, or an expression that evaluates to a list of
  `str`s:
  ```python
  from argtyped import Arguments
  from typing_extensions import Literal

  class MyArgs(Arguments):
      foo: Literal["debug", "info", "warning", "error"]  # 4 choices

  # argv: ["--foo=debug"] => foo="debug"
  ```
  This is equivalent to the `choices` keyword in `argparse.add_argument`.
  
  **Note:** The choice type was previously named `Choices`. This is deprecated in favor of the
  [`Literal` type](https://mypy.readthedocs.io/en/stable/literal_types.html) introduced in Python 3.8 and back-ported to
  3.6 and 3.7 in the `typing_extensions` library. `Choices` was removed since version 0.4.0.
- **Enum types** derived from `enum.Enum`. It is recommended to use `argtyped.Enum` which uses the instance names as
  values:
  ```python
  from argtyped import Enum

  class MyEnum(Enum):
      Debug = auto()    # "debug"
      Info = auto()     # "info"
      Warning = auto()  # "warning"
  ```
- **Switch types** `Switch`. `Switch` arguments are like `bool` arguments, but they don't take values. Instead, a switch
  argument `switch` requires `--switch` to enable and `--no-switch` to disable:
  ```python
  from argtyped import Arguments, Switch

  class MyArgs(Arguments):
      switch: Switch = True
      bool_arg: bool = False

  # argv: []                                 => flag=True,  bool_arg=False
  # argv: ["--switch", "--bool-arg=false"]   => flag=True,  bool_arg=False
  # argv: ["--no-switch", "--bool-arg=true"] => flag=False, bool_arg=True
  # argv: ["--switch=false"]                 => WRONG
  # argv: ["--no-bool-arg"]                  => WRONG
  ```
- **List types** `List[T]`, where `T` is any supported type except switch types. List arguments allow passing multiple
  values on the command line following the argument flag, it is equivalent to setting `nargs="*"` in `argparse`.
  
  Although there is no built-in support for other `nargs` settings such as `"+"` (one or more) or `N` (fixed number),
  you can add custom validation logic by overriding the `__init__` method in your `Arguments` subclass.
- **Optional types** `Optional[T]`, where `T` is any supported type except list or switch types. An optional argument
  will be filled with `None` if no value is provided. It could also be explicitly set to `None` by using `none` as value
  in the command line:
  ```python
  from argtyped import Arguments
  from typing import Optional

  class MyArgs(Arguments):
      opt_arg: Optional[int]  # implicitly defaults to `None`

  # argv: []                 => opt_arg=None
  # argv: ["--opt-arg=1"]    => opt_arg=1
  # argv: ["--opt-arg=none"] => opt_arg=None
  ```
- Any other type that takes a single `str` as `__init__` parameters. It is also theoretically possible to use a function
  that takes an `str` as input, but it's not recommended as it's not type-safe.
  
## Composing `Arguments` Classes

You can split your arguments into separate `Arguments` classes and then compose them together by inheritance. A subclass
will have the union of all arguments in its base classes. If the subclass contains an argument with the same name as an
argument in a base class, then the subclass definition takes precedence. For example:

```python
class BaseArgs(Arguments):
    a: int = 1
    b: Switch = True

class DerivedArgs(BaseArgs):
    b: str

# args = DerivedArgs([])  # bad; `b` is required
args = DerivedArgs(["--b=1234"])
```

**Caveat:** For simplicity, we do not completely follow the [C3 linearization algorithm](
https://en.wikipedia.org/wiki/C3_linearization) that determines the class MRO in Python. Thus, it is a bad idea to have
overridden arguments in cases where there's diamond inheritance.

If you don't understand the above, that's fine. Just note that generally, it's a bad idea to have too complicated
inheritance relationships with overridden arguments.

## Argument Naming Styles

By default `argtyped` uses `--kebab-case` (with hyphens connecting words), which is the convention for UNIX command line
tools. However, many existing tools use the awkward `--snake_case` (with underscores connecting words), and sometimes
consistency is preferred over aesthetics. If you want to use underscores, you can do so by setting `underscore=True`
inside the parentheses where you specify base classes, like this:

```python
class UnderscoreArgs(Arguments, underscore=True):
    underscore_arg: int
    underscore_switch: Switch = True

args = UnderscoreArgs(["--underscore_arg", "1", "--no_underscore_switch"])
```

The underscore settings only affect arguments defined in the class scope; (non-overridden) inherited arguments are not
affects. Thus, you can mix-and-match `snake_case` and `kebab-case` arguments:

```python
class MyArgs(UnderscoreArgs):
    kebab_arg: str

class MyFinalArgs(MyArgs, underscore=True):
    new_underscore_arg: float

args = MyArgs(["--underscore_arg", "1", "--kebab-arg", "kebab", "--new_underscore_arg", "1.0"])
```

## Notes

- Advanced `argparse` features such as subparsers, groups, argument lists, and custom actions are not supported.
- Using switch arguments may result in name clashes: if a switch argument has name `arg`, there can be no argument with
  the name `no_arg`.
- Optional types:
  - `Optional` can be used with `Literal`:
    ```python
    from argtyped import Arguments
    from typing import Literal, Optional
    
    class MyArgs(Arguments):
        foo: Optional[Literal["a", "b"]]  # valid
        bar: Literal["a", "b", "none"]    # also works but is less obvious
    ```
  - `Optional[str]` would parse a value of `"none"` (case-insensitive) into `None`.
- List types:
  - `List[Optional[T]]` is a valid type. For example:
    ```python
    from argtyped import Arguments
    from typing import List, Literal, Optional
    
    class MyArgs(Arguments):
        foo: List[Optional[Literal["a", "b"]]] = ["a", None, "b"]  # valid
    
    # argv: ["--foo", "a", "b", "none", "a", "b"] => foo=["a", "b", None, "a", "b"]
    ```
  - List types cannot be nested inside a list or an optional type. Types such as `Optional[List[int]]` and
    `List[List[int]]` are not accepted.

## Under the Hood

This is what happens under the hood:
1. When a subclass of `argtyped.Arguments` is constructed, type annotations and class-level attributes (i.e., the
   default values) are collected to form argument declarations.
2. After verifying the validity of declared arguments, `argtyped.ArgumentSpec` are created for each argument and stored
   within the subclass as the `__arguments__` class attribute.
3. When an instance of the subclass is initialized, if it's the first time, an instance of `argparse.ArgumentParser` is
   created and arguments are registered with the parser. The parser is cached in the subclass as the `__parser__`
   attribute.
4. The parser's `parse_args` method is invoked with either `sys.argv` or strings provided as parameters, returning
   parsed arguments.
5. The parsed arguments are assigned to `self` (the instance of the `Arguments` subclass being initialized).

## Todo

- [ ] Support `action="append"` or `action="extend"` for `List[T]` types.
  - Technically this is not a problem, but there's no elegant way to configure whether this behavior is desired.
- [ ] Throw (suppressible) warnings on using non-type callables as types.
- [ ] Support converting an `attrs` class into `Arguments`.
- [ ] Support forward references in type annotations.
