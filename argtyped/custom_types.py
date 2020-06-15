import enum
from typing import Any, Iterable, Optional, Tuple, Type, TypeVar, Union

__all__ = [
    "Choices",
    "Enum",
    "auto",  # also export auto for convenience
    "Switch",
    "is_choices",
    "is_enum",
    "is_optional",
    "unwrap_optional",
]

auto = enum.auto

NoneType = type(None)
T = TypeVar('T')


class _Choices:
    def __new__(cls, values=None):
        self = super().__new__(cls)
        self.__values__ = values
        return self

    def __getitem__(self, values: Union[str, Iterable[str]]):
        if isinstance(values, Iterable) and not isinstance(values, str):
            parsed_values = tuple(values)
        else:
            parsed_values = (values,)
        if len(parsed_values) == 0:
            raise TypeError("Choices must contain at least one element")
        return self.__class__(parsed_values)


Choices: Any = _Choices()


class Enum(enum.Enum):
    # pylint: disable=no-self-argument, unused-argument
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()

    # pylint: enable=no-self-argument, unused-argument

    def __eq__(self, other):
        return self.value == other or super().__eq__(other)


# Switch is a type that's different but equivalent to `bool`.
# It is defined as the `Union` of `bool` and a dummy type, because:
# 1. `bool` cannot be sub-typed.
# 2. `Union` with a single (possibly duplicated) type is flattened into that type.
_dummy_type = type("--invalid-type--", (), {})
Switch = Union[bool, _dummy_type]  # type: ignore[valid-type]

HAS_LITERAL = False
_Literal = None
try:
    from typing import Literal  # type: ignore

    HAS_LITERAL = True
except ImportError:
    try:
        from typing_extensions import Literal  # type: ignore

        try:
            from typing_extensions import _Literal  # type: ignore  # compat. with Python 3.6
        except ImportError:
            pass

        HAS_LITERAL = True
    except ImportError:
        pass

if HAS_LITERAL:
    def is_choices(typ: type) -> bool:
        r"""Check whether a type is a choices type (:class:`Choices` or :class:`Literal`). This cannot be checked using
        traditional methods,  since :class:`Choices` is a metaclass.
        """
        return (isinstance(typ, _Choices) or
                getattr(typ, '__origin__', None) is Literal or
                type(typ) is _Literal)  # pylint: disable=unidiomatic-typecheck


    def unwrap_choices(typ: type) -> Tuple[str, ...]:
        r"""Return the string literals associated with the choices type. Literal type in Python 3.7+ stores the literals
        in ``typ.__args__``, but in Python 3.6- it's in ``typ.__values__``.
        """
        return typ.__values__ if hasattr(typ, "__values__") else typ.__args__  # type: ignore[attr-defined]

else:
    def is_choices(typ: type) -> bool:
        r"""Check whether a type is a choices type (:class:`Choices`). This cannot be checked using traditional methods,
        since :class:`Choices` is a metaclass.
        """
        return isinstance(typ, _Choices)


    def unwrap_choices(typ: type) -> Tuple[str, ...]:
        r"""Return the string literals associated with the choices type."""
        return typ.__values__  # type: ignore[attr-defined]


def is_enum(typ: Any) -> bool:
    r"""Check whether a type is an Enum type. Since we're using ``issubclass``, we need to check whether :arg:`typ`
    is a type first."""
    return isinstance(typ, type) and issubclass(typ, enum.Enum)


def is_optional(typ: type) -> bool:
    r"""Check whether a type is `Optional[T]`. `Optional` is internally implemented as `Union` with `type(None)`."""
    return getattr(typ, '__origin__', None) is Union and NoneType in typ.__args__  # type: ignore


def unwrap_optional(typ: Type[Optional[T]]) -> Type[T]:
    r"""Return the inner type inside an `Optional[T]` type."""
    return next(t for t in typ.__args__ if not isinstance(t, NoneType))  # type: ignore
