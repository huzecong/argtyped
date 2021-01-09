import argparse
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
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from .custom_types import (
    Switch,
    is_choices,
    is_enum,
    is_list,
    is_optional,
    unwrap_optional,
    unwrap_list,
    unwrap_choices,
)

__all__ = [
    "Arguments",
    "ArgumentSpec",
    "argument_specs",
]

T = TypeVar("T")
ConversionFn = Callable[[str], T]


class ArgumentParser(argparse.ArgumentParser):
    r""" A class to override some of ``ArgumentParser``\ 's behaviors. """

    def _get_value(self, action, arg_string):
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

    def error(self, message):
        r"""
        The original ``error`` method only prints the usage and force quits. Here we
        print the full help.
        """
        self.print_help(sys.stderr)
        sys.stderr.write(f"{self.prog}: error: {message}\n")
        self.exit(2)

    def add_switch_argument(self, name: str, default: bool = False) -> None:
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
        self.add_argument(f"--no-{name}", action="store_false", dest=var_name)


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


class ArgumentSpec(NamedTuple):  # pylint: disable=inherit-non-class
    # NOTE: pylint raises false-positive error only on Python 3.9. This will likely be
    #  fixed in the next release (2.6.1).
    nullable: bool
    required: bool
    value_type: type
    type: str  # Literal["normal", "switch", "sequence"]
    choices: Optional[Tuple[Any, ...]] = None
    default: Optional[Any] = None


class ArgumentsMeta(ABCMeta):
    r"""
    Metaclass for :class:`Arguments`. The type annotations are parsed and converted into
    an ``argparse.ArgumentParser`` on class creation.
    """
    __parser__: Optional[ArgumentParser]
    __arguments__: "OrderedDict[str, ArgumentSpec]"

    def __new__(mcs, name, bases, namespace, **kwargs):
        cls: "Type[Arguments]" = super().__new__(mcs, name, bases, namespace)

        root = kwargs.get("_root", False)
        if not root and not issubclass(cls, Arguments):
            raise TypeError(f"Type {cls.__name__!r} must inherit from `Arguments`")
        if root:
            cls.__parser__ = None
            cls.__arguments__ = {}
            return cls

        arguments: "OrderedDict[str, ArgumentSpec]" = OrderedDict()
        for base in reversed(bases):
            # Use reversed order so derived classes can override base annotations.
            if issubclass(base, Arguments):
                arguments.update(argument_specs(base))

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

        def type_error(message: str) -> None:
            # pylint: disable=undefined-loop-variable
            raise TypeError(
                f"Argument {arg_name!r} has invalid type {annotations[arg_name]!r}: "
                + message
            )

        def value_error(message: str) -> None:
            # pylint: disable=undefined-loop-variable
            raise TypeError(
                f"Argument {arg_name!r} has invalid default value {default_val!r}: "
                + message
            )

        # Check validity of arguments and create specs.
        for arg_name, arg_typ in annotations.items():
            if isinstance(arg_typ, str):
                type_error("forward references are not yet supported")

            has_default = arg_name in namespace
            default_val = namespace.get(arg_name, None)

            # On sequence and nullable types:
            # - Nested lists (e.g. `List[List[T]]`) are not supported.
            # - When mixing `List` and `Optional`, the only allowed variant is
            #   `List[Optional[T]]`. Anything else (`Optional[List[T]]`,
            #   `Optional[List[Optional[T]]]`) are invalid.
            sequence = is_list(arg_typ)
            if sequence:
                arg_typ = unwrap_list(arg_typ)
            nullable = is_optional(arg_typ)
            if nullable:
                arg_typ = unwrap_optional(arg_typ)
            if (sequence or nullable) and (arg_typ is Switch or is_list(arg_typ)):
                type_error(
                    f"{'List' if is_list(arg_typ) else 'Switch'!r} cannot be "
                    f"nested inside {'List' if sequence else 'Optional'!r}",
                )

            required = False
            if not has_default:
                if nullable and not sequence:
                    has_default = True
                    default_val = None
                elif not nullable:
                    required = True

            if not nullable and has_default and default_val is None:
                value_error(
                    "Argument not nullable. Change type annotation to 'Optional[...]' "
                    "to allow values of None"
                )

            if arg_typ is Switch:
                if not isinstance(default_val, bool):
                    value_error("Switch argument must have a boolean default value")
                spec = ArgumentSpec(
                    type="switch",
                    nullable=False,
                    required=False,
                    value_type=bool,
                    default=default_val,
                )
            else:
                if sequence and has_default and not isinstance(default_val, list):
                    value_error("Default for list argument must be of list type")
                if is_enum(arg_typ) or is_choices(arg_typ):
                    if is_enum(arg_typ):
                        value_type = arg_typ
                        choices = tuple(arg_typ)
                    else:
                        value_type = str
                        choices = unwrap_choices(arg_typ)
                        if any(not isinstance(choice, str) for choice in choices):
                            type_error("All choices must be strings")
                    if has_default:
                        if not all(
                            x in choices or (x is None and nullable)
                            for x in (default_val if sequence else [default_val])
                        ):
                            value_error("Value must be among valid choices")
                else:
                    if arg_typ not in _TYPE_CONVERSION_FN and not callable(arg_typ):
                        type_error("Unsupported type")
                    value_type = arg_typ
                    choices = None
                spec = ArgumentSpec(
                    type="sequence" if sequence else "normal",
                    nullable=nullable,
                    required=required,
                    value_type=value_type,
                    choices=choices,
                    default=default_val,
                )
            arguments[arg_name] = spec

        # The parser will be lazily constructed when the `Arguments` instance is first
        # initialized.
        cls.__parser__ = None
        cls.__arguments__ = arguments
        return cls

    def build_parser(cls) -> ArgumentParser:
        parser = ArgumentParser()
        for name, spec in cls.__arguments__.items():
            arg_name = "--" + name.replace("_", "-")
            if spec.type in {"normal", "sequence"}:
                arg_type = spec.value_type
                conversion_fn = _TYPE_CONVERSION_FN.get(arg_type, arg_type)
                if spec.nullable:
                    conversion_fn = _optional_wrapper_fn(conversion_fn)
                kwargs: Dict[str, Any] = {
                    "required": spec.required,
                    "type": conversion_fn,
                }
                if spec.choices is not None:
                    if spec.nullable:
                        kwargs["choices"] = spec.choices + (None,)
                    else:
                        kwargs["choices"] = spec.choices
                    if is_enum(spec.value_type):
                        # Display only the enum names in help.
                        kwargs["metavar"] = (
                            "{" + ",".join(val.name for val in spec.choices) + "}"
                        )
                if not spec.required:
                    kwargs["default"] = spec.default
                if spec.type == "sequence":
                    kwargs["nargs"] = "*"
                parser.add_argument(arg_name, **kwargs)
            else:
                assert spec.default is not None
                parser.add_switch_argument(arg_name, spec.default)

        if cls.__module__ != "__main__":
            # Usually arguments are defined in the same script that is directly
            # run (__main__). If this is not the case, add a note in help message
            # indicating where the arguments are defined.
            parser.epilog = (
                f"Note: Arguments defined in {cls.__module__}.{cls.__name__}"
            )
        return parser


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
        if self.__class__.__parser__ is None:
            self.__class__.__parser__ = self.__class__.build_parser()
        namespace = self.__class__.__parser__.parse_args(args)
        for arg_name in argument_specs(self.__class__):
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
        valid_keys = list(argument_specs(self.__class__).keys())
        valid_vals = [repr(getattr(self, k)) for k in valid_keys]
        max_key = max(len(k_col), max(len(k) for k in valid_keys))
        max_val = max(len(v_col), max(len(v) for v in valid_vals))
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
        for k, v in zip(valid_keys, valid_vals):
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
