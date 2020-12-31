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
    Literal,
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
    is_optional,
    unwrap_optional,
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


class ArgumentSpec(NamedTuple):
    nullable: bool
    required: bool
    value_type: Union[type, ConversionFn[Any]]
    type: Literal["normal", "switch"]
    choices: Optional[Tuple[str, ...]] = None
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

        # Check validity of arguments and create specs.
        for arg_name, arg_typ in annotations.items():
            has_default = arg_name in namespace
            default_val = namespace.get(arg_name, None)
            nullable = is_optional(arg_typ)
            if nullable:
                # extract the type wrapped inside `Optional`
                arg_typ = unwrap_optional(arg_typ)

            required = False
            if nullable and not has_default:
                has_default = True
                default_val = None
            elif not nullable and not has_default:
                required = True

            if not nullable and has_default and default_val is None:
                raise TypeError(
                    f"Argument {arg_name!r} has default value of None, but is not "
                    f"nullable. Change type annotation to `Optional[...]` to allow "
                    f"values of `None`"
                )

            if arg_typ is Switch:  # type: ignore[misc]
                if not isinstance(default_val, bool):
                    raise TypeError(
                        f"Switch argument {arg_name!r} must have a default value of "
                        f"type bool"
                    )
                spec = ArgumentSpec(
                    type="switch",
                    nullable=False,
                    required=False,
                    value_type=bool,
                    default=default_val,
                )
            else:
                if is_enum(arg_typ):
                    value_type = arg_typ
                    choices = tuple(x.name for x in arg_typ)
                    if has_default and not isinstance(default_val, arg_typ):
                        raise TypeError(
                            f"Invalid default value for argument {arg_name!r}"
                        )
                elif is_choices(arg_typ):
                    value_type = str
                    choices = unwrap_choices(arg_typ)
                    if any(not isinstance(choice, str) for choice in choices):
                        raise TypeError("All choices must be strings")
                    if has_default and default_val not in choices:
                        raise TypeError(
                            f"Invalid default value for argument {arg_name!r}"
                        )
                else:
                    if arg_typ not in _TYPE_CONVERSION_FN and not callable(arg_typ):
                        raise TypeError(
                            f"Invalid type {arg_typ!r} for argument {arg_name!r}"
                        )
                    value_type = arg_typ
                    choices = None
                spec = ArgumentSpec(
                    type="normal",
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

    def build_parser(cls):
        parser = ArgumentParser()
        for name, spec in cls.__arguments__.items():
            arg_name = "--" + name.replace("_", "-")
            if spec.type == "normal":
                arg_type = spec.value_type
                conversion_fn = _TYPE_CONVERSION_FN.get(arg_type, arg_type)
                if spec.nullable:
                    conversion_fn = _optional_wrapper_fn(conversion_fn)
                kwargs = {
                    "required": spec.required,
                    "type": conversion_fn,
                }
                if not spec.required:
                    kwargs["default"] = spec.default
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
        cls.__parser__ = parser


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
        if self.__parser__ is None:
            self.__class__.build_parser()
        namespace = self.__parser__.parse_args(args)
        for arg_name in self.__arguments__:
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


def argument_specs(args_class: Type[Arguments]) -> "OrderedDict[str, ArgumentSpec]":
    r"""
    Return a dictionary mapping argument names to their specs (:class:`ArgumentSpec`
    objects).
    """
    return args_class.__arguments__
