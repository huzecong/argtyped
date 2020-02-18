import argparse
import functools
import sys
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, OrderedDict as OrderedDictT, TypeVar

from .custom_types import Switch, is_choices, is_enum, is_optional, unwrap_optional

__all__ = [
    "Arguments",
]

T = TypeVar('T')
ConversionFn = Callable[[str], T]


class ArgumentParser(argparse.ArgumentParser):
    r"""A class to override some of ``ArgumentParser``\ 's behaviors.
    """

    def _get_value(self, action, arg_string):
        r"""The original ``_get_value`` method catches exceptions in user-defined ``type_func``\ s and ignores the
        error message. Here we don't do that.
        """
        type_func = self._registry_get('type', action.type, action.type)

        try:
            result = type_func(arg_string)
        except (argparse.ArgumentTypeError, TypeError, ValueError) as e:
            message = f"value '{arg_string}', {e.__class__.__name__}: {str(e)}"
            raise argparse.ArgumentError(action, message)

        return result

    def error(self, message):
        r"""The original ``error`` method only prints the usage and force quits. Here we print the full help.
        """
        self.print_help(sys.stderr)
        sys.stderr.write(f"{self.prog}: error: {message}\n")
        self.exit(2)

    def add_switch_argument(self, name: str, default: bool = False) -> None:
        r"""Add a "switch" argument to the parser. A switch argument with name ``"flag"`` has value ``True`` if the
        argument ``--flag`` exists, and ``False`` if ``--no-flag`` exists.
        """
        assert name.startswith("--")
        name = name[2:]
        var_name = name.replace('-', '_')
        self.add_argument(f"--{name}", action="store_true", default=default, dest=var_name)
        self.add_argument(f"--no-{name}", action="store_false", dest=var_name)


def _bool_conversion_fn(s: str) -> bool:
    if s.lower() in ["y", "yes", "true", "ok"]:
        return True
    if s.lower() in ["n", "no", "false"]:
        return False
    raise ValueError(f"Invalid value '{s}' for bool argument")


def _optional_wrapper_fn(fn: ConversionFn[T]) -> ConversionFn[Optional[T]]:
    @functools.wraps(fn)
    def wrapped(s: str) -> Optional[T]:
        if s.lower() == 'none':
            return None
        return fn(s)

    return wrapped


_TYPE_CONVERSION_FN: Dict[type, ConversionFn[Any]] = {
    bool: _bool_conversion_fn,
}


class Arguments:
    r"""A typed version of ``argparse``. It's easier to illustrate using an example:

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
        parser.add_argument("--activation", choices=["relu", "tanh", "sigmoid"], default="relu")
        parser.add_argument("--logging-level", choices=ghcc.logging.get_levels(), default="info")
        parser.add_argument("--use-dropout", action="store_true", dest="use_dropout", default=True)
        parser.add_argument("--no-use-dropout", action="store_false", dest="use_dropout")
        parser.add_argument("--dropout-prob", type=lambda s: None if s.lower() == 'none' else float(s), default=0.5)

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

        Namespace(model_name="LSTM", hidden_size=512, activation="sigmoid", logging_level="debug",
                  use_dropout=False, dropout_prob=None)

    :class:`Arguments` provides the following features:

    - More concise and intuitive syntax over ``argparse``, less boilerplate code.
    - Arguments take the form of type-annotated class attributes, allowing IDEs to provide autocompletion.
    - Drop-in replacement for ``argparse``, since internally ``argparse`` is used.

    **Note:** Advanced features such as subparsers, groups, argument lists, custom actions are not supported.
    """

    _annotations: OrderedDictT[str, type]

    def __init__(self, args: Optional[List[str]] = None):
        annotations: OrderedDictT[str, type] = OrderedDict()
        for base in reversed(self.__class__.mro()):
            # Use reversed order so derived classes can override base annotations.
            if base not in [object, Arguments]:
                annotations.update(base.__dict__.get('__annotations__', {}))

        # Check if there are arguments with default values but without annotations.
        for key in dir(self):
            value = getattr(self, key)
            if not key.startswith("__") and not callable(value):
                if key not in annotations:
                    raise ValueError(f"Argument '{key}' does not have type annotation")

        parser = ArgumentParser()
        for arg_name, arg_typ in annotations.items():
            # Check validity of name and type.

            has_default = hasattr(self.__class__, arg_name)
            default_val = getattr(self.__class__, arg_name, None)
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
                raise ValueError(f"Argument '{arg_name}' has default value of None, but is not nullable")

            parser_arg_name = "--" + arg_name.replace("_", "-")
            parser_kwargs: Dict[str, Any] = {
                "required": required,
            }
            if arg_typ is Switch:  # type: ignore
                if not isinstance(default_val, bool):
                    raise ValueError(f"Switch argument '{arg_name}' must have a default value of type bool")
                parser.add_switch_argument(parser_arg_name, default_val)
            elif is_choices(arg_typ) or is_enum(arg_typ):
                if is_enum(arg_typ):
                    choices = list(arg_typ)  # type: ignore
                    parser_kwargs["type"] = arg_typ
                else:
                    choices = arg_typ.__values__  # type: ignore
                parser_kwargs["choices"] = choices
                if has_default:
                    if default_val not in choices:
                        raise ValueError(f"Invalid default value for argument '{arg_name}'")
                    parser_kwargs["default"] = default_val
                parser.add_argument(parser_arg_name, **parser_kwargs)
            else:
                if arg_typ not in _TYPE_CONVERSION_FN and not callable(arg_typ):
                    raise ValueError(f"Invalid type '{arg_typ}' for argument '{arg_name}'")
                conversion_fn = _TYPE_CONVERSION_FN.get(arg_typ, arg_typ)
                if nullable:
                    conversion_fn = _optional_wrapper_fn(conversion_fn)
                parser_kwargs["type"] = conversion_fn
                if has_default:
                    parser_kwargs["default"] = default_val
                parser.add_argument(parser_arg_name, **parser_kwargs)

        if self.__class__.__module__ != "__main__":
            # Usually arguments are defined in the same script that is directly run (__main__).
            # If this is not the case, add a note in help message indicating where the arguments are defined.
            parser.epilog = f"Note: Arguments defined in {self.__class__.__module__}.{self.__class__.__name__}"

        namespace = parser.parse_args(args)
        self._annotations = annotations
        for arg_name, arg_typ in annotations.items():
            setattr(self, arg_name, getattr(namespace, arg_name))

    def to_dict(self) -> OrderedDictT[str, Any]:
        r"""Convert the set of arguments to a dictionary.

        :return: An ``OrderedDict`` mapping argument names to values.
        """
        return OrderedDict([(key, getattr(self, key)) for key in self._annotations.keys()])

    def to_string(self, width: Optional[int] = None, max_width: Optional[int] = None) -> str:
        r"""Represent the arguments as a table.

        :param width: Width of the printed table. Defaults to ``None``, which fits the table to its contents. An
            exception is raised when the table cannot be drawn with the given width.
        :param max_width: Maximum width of the printed table. Defaults to ``None``, meaning no limits. Must be ``None``
            if :arg:`width` is not ``None``.
        """
        if width is not None and max_width is not None:
            raise ValueError("`max_width` must be None when `width` is specified")

        k_col = "Arguments"
        v_col = "Values"
        valid_keys = list(self._annotations.keys())
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
                v = v[:((max_val - 5) // 2)] + ' ... ' + v[-((max_val - 4) // 2):]
                assert len(v) == max_val
            return f"║ {k.ljust(max_key)} │ {v.ljust(max_val)} ║\n"

        s = repr(self.__class__) + '\n'
        s += f"╔═{'═' * max_key}═╤═{'═' * max_val}═╗\n"
        s += get_row(k_col, v_col)
        s += f"╠═{'═' * max_key}═╪═{'═' * max_val}═╣\n"
        for k, v in zip(valid_keys, valid_vals):
            s += get_row(k, v)
        s += f"╚═{'═' * max_key}═╧═{'═' * max_val}═╝\n"
        return s
