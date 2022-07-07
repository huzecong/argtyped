import argparse
import enum
import functools
import shutil
import sys
from abc import ABCMeta
from collections import OrderedDict
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    NoReturn,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from argtyped.custom_types import (
    Switch,
    is_choices,
    is_enum,
    is_list,
    is_optional,
    unwrap_choices,
    unwrap_list,
    unwrap_optional,
)

__all__ = [
    "Arguments",
    "ArgumentKind",
    "ArgumentSpec",
    "argument_specs",
]

T = TypeVar("T")
ConversionFn = Callable[[str], T]


class ArgumentParser(argparse.ArgumentParser):
    r"""A class to override some of ``ArgumentParser``\ 's behaviors."""

    def _get_value(self, action: argparse.Action, arg_string: str) -> Any:
        r"""
        The original ``_get_value`` method catches exceptions in user-defined
        ``type_func``\ s and ignores the error message. Here we don't do that.
        """
        type_func = self._registry_get("type", action.type, action.type)

        try:
            result = type_func(arg_string)
        except (argparse.ArgumentTypeError, TypeError, ValueError) as e:
            message = f"value '{arg_string}', {e.__class__.__name__}: {str(e)}"
            raise argparse.ArgumentError(action, message)

        return result

    def error(self, message: str) -> NoReturn:
        r"""
        The original ``error`` method only prints the usage and force quits. Here we
        print the full help.
        """
        self.print_help(sys.stderr)
        sys.stderr.write(f"{self.prog}: error: {message}\n")
        self.exit(2)

    def add_switch_argument(
        self, name: str, default: bool = False, underscore: bool = False
    ) -> None:
        r"""
        Add a "switch" argument to the parser. A switch argument with name ``"flag"``
        has value ``True`` if the argument ``--flag`` exists, and ``False`` if
        ``--no-flag`` exists.
        """
        assert name.startswith("--")
        name = name[2:]
        var_name = name.replace("-", "_")
        self.add_argument(
            f"--{name}", action="store_true", default=default, dest=var_name
        )
        off_arg_name = f"--no_{name}" if underscore else f"--no-{name}"
        self.add_argument(off_arg_name, action="store_false", dest=var_name)


def _bool_conversion_fn(s: str) -> bool:
    if s.lower() in {"y", "yes", "true", "ok"}:
        return True
    if s.lower() in {"n", "no", "false"}:
        return False
    raise ValueError(f"Invalid value {s!r} for bool argument")


def _optional_wrapper_fn(fn: ConversionFn[T]) -> ConversionFn[Optional[T]]:
    @functools.wraps(fn)
    def wrapped(s: str) -> Optional[T]:
        if s.lower() == "none":
            return None
        return fn(s)

    return wrapped


_TYPE_CONVERSION_FN: Dict[type, ConversionFn[Any]] = {
    bool: _bool_conversion_fn,
}


class ArgumentKind(enum.Enum):
    """
    The kind of argument:

    - ``NORMAL``: A normal argument that takes a single value.
    - ``SWITCH``: A boolean switch argument that takes no values; it is set to True with
      ``--argument`` and False with ``--no-argument``
      (``action="store_true"/"store_false"``).
    - ``SEQUENCE``: A sequential argument that takes multiple values (``nargs="*"``).
    """

    NORMAL = 0
    SWITCH = 1
    SEQUENCE = 2


_NOTHING = object()  # sentinel


class ArgumentSpec(NamedTuple):
    """
    Internal specs of an argument.

    This class is internal -- there's no stability guarantees on its attributes across
    versions.
    """

    name: str
    nullable: bool
    required: bool
    type: type
    kind: ArgumentKind
    choices: Optional[Tuple[Any, ...]] = None
    default: Any = _NOTHING

    positional: bool = False
    # ^ Whether the argument is a positional argument.  If False, it is a keyword(-only)
    #   argument.
    argparse_options: Optional[Dict[str, Any]] = None
    # ^ Additional arguments to pass to `ArgumentParser.add_argument`.  This takes
    #   precedence over `argtyped`'s computed options, e.g. you can set `nargs="+"` for
    #   sequence-type arguments.
    parse: bool = True
    # ^ Whether the argument value should be parsed.  If False, it is the downstream's
    #   responsibility to parse (e.g. in `attrs`).
    underscore: bool = False
    # ^ Argument naming convention:
    #   True for `--snake_case` args, False for `--kebab-case` (default).
    inherited: bool = False
    # ^ True if argument was defined in a base class and is not overridden in the
    #   current class.

    def with_options(self, positional: bool = False, **kwargs: Any) -> "ArgumentSpec":
        """Return a new spec with additional ``argparse`` options."""
        return self._replace(  # pylint: disable=no-member
            positional=positional, argparse_options=kwargs or None
        )

    def with_default(self, value: Optional[Any]) -> "ArgumentSpec":
        """
        Return a new spec with a default value, and perform validation on the value.

        By default, we don't store default values on the spec.  This is because default
        value handling could happen outside ``argtyped``, e.g. for
        :class:`AttrsArguments` it is handled by ``attrs``.
        """

        def value_error(message: str) -> NoReturn:
            raise TypeError(
                f"Argument {self.name!r} has invalid default value {value!r}: {message}"
            )

        if not self.nullable and value is None:
            value_error(
                "Argument not nullable. Change type annotation to 'Optional[...]' "
                "to allow values of None"
            )
        if self.kind == ArgumentKind.SWITCH:
            if not isinstance(value, bool):
                value_error("Switch argument must have a boolean default value")
        if self.kind == ArgumentKind.SEQUENCE:
            if not isinstance(value, list):
                value_error("Default for list argument must be of list type")
            value_seq = value
        else:
            value_seq = [value]
        if self.choices is not None:
            if not all(
                x in self.choices  # pylint: disable=unsupported-membership-test
                or (x is None and self.nullable)
                for x in value_seq
            ):
                value_error("Value must be among valid choices")

        return self._replace(default=value, required=False)  # pylint: disable=no-member


def _generate_argument_spec(
    arg_name: str,
    arg_type: Any,
    has_default: bool,
    *,
    underscore: bool = False,
) -> ArgumentSpec:
    original_type = arg_type

    def type_error(message: str) -> None:
        raise TypeError(
            f"Argument {arg_name!r} has invalid type {original_type!r}: {message}"
        )

    if isinstance(arg_type, str):
        type_error("forward references are not yet supported")

    # On sequence and nullable types:
    # - Nested lists (e.g. `List[List[T]]`) are not supported.
    # - When mixing `List` and `Optional`, the only allowed variant is
    #   `List[Optional[T]]`. Anything else (`Optional[List[T]]`,
    #   `Optional[List[Optional[T]]]`) is invalid.
    sequence = is_list(arg_type)
    if sequence:
        arg_type = unwrap_list(arg_type)
    nullable = is_optional(arg_type)
    if nullable:
        arg_type = unwrap_optional(arg_type)
    if (sequence or nullable) and (arg_type is Switch or is_list(arg_type)):
        type_error(
            f"{'List' if is_list(arg_type) else 'Switch'!r} cannot be "
            f"nested inside {'List' if sequence else 'Optional'!r}",
        )

    if arg_type is Switch:
        return ArgumentSpec(
            name=arg_name,
            kind=ArgumentKind.SWITCH,
            required=False,
            nullable=False,
            type=bool,
            underscore=underscore,
        )

    choices = None
    if is_enum(arg_type) or is_choices(arg_type):
        if is_enum(arg_type):
            value_type = arg_type
            choices = tuple(arg_type)
        else:
            value_type = str
            choices = unwrap_choices(arg_type)
            if any(not isinstance(choice, str) for choice in choices):
                type_error("All choices must be strings")
    else:
        if arg_type not in _TYPE_CONVERSION_FN and not callable(arg_type):
            type_error("Unsupported type")
        value_type = arg_type
    return ArgumentSpec(
        name=arg_name,
        kind=ArgumentKind.SEQUENCE if sequence else ArgumentKind.NORMAL,
        required=not has_default,
        nullable=nullable,
        type=value_type,
        choices=choices,
        underscore=underscore,
    )


def _build_parser(
    arguments: "OrderedDict[str, ArgumentSpec]", cls: type
) -> ArgumentParser:
    """Create the :class:`ArgumentParser` for this :class:`Arguments` class."""
    parser = ArgumentParser()
    for name, spec in arguments.items():
        arg_name = name if spec.underscore else name.replace("_", "-")
        if not spec.positional:
            arg_name = f"--{arg_name}"
        if spec.kind in {ArgumentKind.NORMAL, ArgumentKind.SEQUENCE}:
            arg_type = spec.type
            kwargs: Dict[str, Any] = {}
            if spec.positional:
                if not spec.required:
                    kwargs["nargs"] = "?"
                    kwargs["default"] = spec.default
            else:
                if spec.required:
                    kwargs["required"] = True
                else:
                    kwargs["default"] = spec.default
            if spec.parse:
                conversion_fn = _TYPE_CONVERSION_FN.get(arg_type, arg_type)
                if spec.nullable:
                    conversion_fn = _optional_wrapper_fn(conversion_fn)
                kwargs["type"] = conversion_fn
            if spec.choices is not None:
                if spec.nullable:
                    kwargs["choices"] = spec.choices + (None,)
                else:
                    kwargs["choices"] = spec.choices
                if is_enum(spec.type):
                    # Display only the enum names in help.
                    kwargs["metavar"] = (
                        "{" + ",".join(val.name for val in spec.choices) + "}"
                    )
            if spec.kind == ArgumentKind.SEQUENCE:
                kwargs["nargs"] = "*"
            if spec.argparse_options is not None:
                kwargs.update(spec.argparse_options)
            parser.add_argument(arg_name, **kwargs)
        else:
            assert spec.default is not None
            parser.add_switch_argument(arg_name, spec.default, spec.underscore)

    if cls.__module__ != "__main__":
        # Usually arguments are defined in the same script that is directly
        # run (__main__). If this is not the case, add a note in help message
        # indicating where the arguments are defined.
        parser.epilog = f"Note: Arguments defined in {cls.__module__}.{cls.__name__}"
    return parser


class ArgumentsMeta(ABCMeta):
    r"""
    Metaclass for :class:`Arguments`. The type annotations are parsed and converted into
    an ``argparse.ArgumentParser`` on class creation.
    """
    __parser__: ArgumentParser
    __arguments__: "OrderedDict[str, ArgumentSpec]"

    def __new__(  # type: ignore[misc]
        mcs,
        name: str,
        bases: Tuple[type, ...],
        namespace: Dict[str, Any],
        **kwargs: Any,
    ) -> "Type[Arguments]":
        cls: "Type[Arguments]" = super().__new__(  # type: ignore[assignment]
            mcs, name, bases, namespace
        )

        root = kwargs.get("_root", False)
        if not root and not issubclass(cls, Arguments):
            raise TypeError(f"Type {cls.__name__!r} must inherit from `Arguments`")
        if root:
            cls.__arguments__ = OrderedDict()
            return cls

        arguments: "OrderedDict[str, ArgumentSpec]" = OrderedDict()
        for base in reversed(cls.mro()[1:]):
            # Use reversed order so derived classes can override base annotations.
            if issubclass(base, Arguments):
                for arg_name, spec in argument_specs(base).items():
                    if spec.inherited:
                        # Skip inherited attributes -- they should have been included
                        # already when we processed the base class, which is higher up
                        # in the MRO.
                        continue
                    arguments[arg_name] = spec._replace(inherited=True)

        # Check if there are arguments with default values but without annotations.
        annotations = getattr(cls, "__annotations__", {})
        for key in annotations:
            if key.startswith("_") and not key.startswith("__"):
                raise TypeError(f"Argument {key!r} must not start with an underscore")
        annotations = OrderedDict(
            [(k, v) for k, v in annotations.items() if not k.startswith("__")]
        )
        for key, value in namespace.items():
            if not key.startswith("_") and not callable(value):
                if key not in annotations and key not in arguments:
                    raise TypeError(f"Argument {key!r} does not have type annotation")

        # Check validity of arguments and create specs.
        underscore = kwargs.get("underscore", False)
        for arg_name, arg_type in annotations.items():
            has_default = arg_name in namespace
            spec = _generate_argument_spec(
                arg_name, arg_type, has_default, underscore=underscore
            )
            if has_default:
                spec = spec.with_default(namespace[arg_name])
            elif spec.kind == ArgumentKind.NORMAL and spec.nullable:
                spec = spec.with_default(None)
            arguments[arg_name] = spec

        # The parser will be lazily constructed when the `Arguments` instance is first
        # initialized.
        cls.__arguments__ = arguments
        return cls


class Arguments(metaclass=ArgumentsMeta, _root=True):
    r"""
    A typed version of ``argparse``. It's easier to illustrate using an example:

    .. code-block:: python

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
            model_name: str
            hidden_size: int = 512
            activation: Choices['relu', 'tanh', 'sigmoid'] = 'relu'
            logging_level: LoggingLevels = LoggingLevels.Info
            use_dropout: Switch = True
            dropout_prob: Optional[float] = 0.5

        args = Arguments()

    This is equivalent to the following code with Python built-in ``argparse``:

    .. code-block:: python

        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--model-name", type=str, required=True)
        parser.add_argument("--hidden-size", type=int, default=512)
        parser.add_argument(
            "--activation", choices=["relu", "tanh", "sigmoid"], default="relu"
        )
        parser.add_argument(
            "--logging-level", choices=ghcc.logging.get_levels(), default="info"
        )
        parser.add_argument(
            "--use-dropout", action="store_true", dest="use_dropout", default=True
        )
        parser.add_argument(
            "--no-use-dropout", action="store_false", dest="use_dropout"
        )
        parser.add_argument(
            "--dropout-prob",
            type=lambda s: None if s.lower() == 'none' else float(s),
            default=0.5
        )

        args = parser.parse_args()

    Suppose the following arguments are provided:

    .. code-block:: bash

        python main.py \
            --model-name LSTM \
            --activation sigmoid \
            --logging-level debug \
            --no-use-dropout \
            --dropout-prob none

    the parsed arguments will be:

    .. code-block:: bash

        Namespace(model_name="LSTM", hidden_size=512, activation="sigmoid",
                  logging_level="debug", use_dropout=False, dropout_prob=None)

    :class:`Arguments` provides the following features:

    - More concise and intuitive syntax over ``argparse``, less boilerplate code.
    - Arguments take the form of type-annotated class attributes, allowing IDEs to
      provide autocompletion.
    - Drop-in replacement for ``argparse``, since internally ``argparse`` is used.

    **Note:** Advanced features such as subparsers, groups, argument lists, custom
    actions are not supported.
    """

    def __init__(self, args: Optional[List[str]] = None):
        cls = self.__class__
        if not hasattr(cls, "__parser__"):
            parser = _build_parser(cls.__arguments__, cls)

            cls.__parser__ = parser
        namespace = cls.__parser__.parse_args(args)
        for arg_name in argument_specs(cls):
            setattr(self, arg_name, getattr(namespace, arg_name))

    def to_dict(self) -> "OrderedDict[str, Any]":
        r"""
        Convert the set of arguments to a dictionary.

        :return: An ``OrderedDict`` mapping argument names to values.
        """
        return OrderedDict(
            [(key, getattr(self, key)) for key in argument_specs(self.__class__).keys()]
        )

    def to_string(
        self, width: Optional[int] = None, max_width: Optional[int] = None
    ) -> str:
        r"""
        Represent the arguments as a table.

        :param width: Width of the printed table. Defaults to ``None``, which fits the
            table to its contents. An exception is raised when the table cannot be drawn
            with the given width.
        :param max_width: Maximum width of the printed table. Defaults to ``None``,
            meaning no limits. Must be ``None`` if :arg:`width` is not ``None``.
        """
        if width is not None and max_width is not None:
            raise ValueError("`max_width` must be None when `width` is specified")

        k_col = "Arguments"
        v_col = "Values"
        arg_reprs = {k: repr(v) for k, v in self.to_dict().items()}
        max_key = max(len(k_col), max(len(k) for k in arg_reprs.keys()))
        max_val = max(len(v_col), max(len(v) for v in arg_reprs.values()))
        margin_col = 7  # table frame & spaces
        if width is not None:
            max_val = width - max_key - margin_col
        elif max_width is not None:
            max_val = min(max_val, max_width - max_key - margin_col)
        if max_val < len(v_col):
            raise ValueError("Table cannot be drawn under the width constraints")

        def get_row(k: str, v: str) -> str:
            if len(v) > max_val:
                v = v[: ((max_val - 5) // 2)] + " ... " + v[-((max_val - 4) // 2) :]
                assert len(v) == max_val
            return f"║ {k.ljust(max_key)} │ {v.ljust(max_val)} ║\n"

        s = repr(self.__class__) + "\n"
        s += f"╔═{'═' * max_key}═╤═{'═' * max_val}═╗\n"
        s += get_row(k_col, v_col)
        s += f"╠═{'═' * max_key}═╪═{'═' * max_val}═╣\n"
        for k, v in arg_reprs.items():
            s += get_row(k, v)
        s += f"╚═{'═' * max_key}═╧═{'═' * max_val}═╝\n"
        return s

    def __repr__(self) -> str:
        columns, _ = shutil.get_terminal_size()
        return self.to_string(max_width=columns)


def argument_specs(
    args_class: Union[Arguments, Type[Arguments]]
) -> "OrderedDict[str, ArgumentSpec]":
    r"""
    Return a dictionary mapping argument names to their specs (:class:`ArgumentSpec`
    objects).
    """
    if isinstance(args_class, Arguments):
        return args_class.__class__.__arguments__
    return args_class.__arguments__
