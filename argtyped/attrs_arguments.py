import argparse
from collections import OrderedDict
from typing import TYPE_CHECKING, ClassVar, List, Optional, Type, TypeVar

from argtyped.arguments import _NOTHING, _build_parser, _generate_argument_spec

__all__ = ["positional_arg", "AttrsArguments"]

if TYPE_CHECKING:
    import attr

    positional_arg = attr.ib
else:

    def positional_arg(*args, **kwargs):
        """Declare an attrs attribute that will become a positional argument."""
        import attr  # pylint: disable=import-outside-toplevel

        metadata = kwargs.get("metadata", {})
        kwargs["metadata"] = {**metadata, "positional": True}
        return attr.ib(*args, **kwargs)


TArgs = TypeVar("TArgs", bound="AttrsArguments")


class AttrsArguments:
    """
    A typed version of ``argparse`` that works for ``attrs`` classes.
    """

    __parser__: ClassVar[argparse.ArgumentParser]

    @classmethod
    def parse_args(cls: Type[TArgs], args: Optional[List[str]] = None) -> TArgs:
        """
        Parse arguments and create an instance of this class.  If ``args`` is not
        specified, then ``sys.argv`` is used.
        """
        import attr  # pylint: disable=import-outside-toplevel

        if not hasattr(cls, "__parser__"):
            # Build argument specs from attrs attributes.
            specs = OrderedDict()
            annotations = getattr(cls, "__annotations__", {})
            for arg_name, attribute in attr.fields_dict(cls).items():
                if not attribute.init:
                    continue  # skip `init=False` attributes
                if arg_name not in annotations:
                    raise TypeError(
                        f"Argument {arg_name!r} does not have type annotation"
                    )
                arg_type = annotations[arg_name]
                has_default = attribute.default is not attr.NOTHING
                spec = _generate_argument_spec(arg_name, arg_type, has_default)
                if attribute.metadata:
                    spec = spec.with_options(**attribute.metadata)
                if attribute.converter is not None:
                    # Do not parse values for attributes with custom converters.
                    spec = spec._replace(parse=False)
                specs[arg_name] = spec
            parser = _build_parser(specs, cls)
            cls.__parser__ = parser
        namespace = cls.__parser__.parse_args(args)
        # Unfilled values will have a special sentinel value; filter them out and let
        # `attrs` fill in defaults.
        values = {k: v for k, v in vars(namespace).items() if v is not _NOTHING}
        return cls(**values)
