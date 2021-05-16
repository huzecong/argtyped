# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Support for `snake_case` (underscore style) arguments. (#6)
- A `DeprecationWarning` is now shown when using `Choices` instead of `Literal`.

## [0.3.0]
### Added
- Support for list arguments. (#1)
- `__repr__` method for `Arguments`. (#2)
- Defined arguments are now stored in a special class variable `__arguments__` in the `Arguments` subclass namespace.
  A utility function `argtyped.argument_specs(Args)` is provided to inspect the specifications of the arguments.

### Changed
- Annotations are now parsed on class construction, and the `argparse.ArgumentParser` object is stored for the whole
  lifetime of the class as `__parser__`.
- Exceptions thrown for invalid type annotations are changed from `ValueError`s to `TypeError`s.

### Fixed
- It is now possible to override an argument with default value defined in the base class with a new argument that does
  not have a default. Namely, the following code is now valid (although discouraged):
  ```python
  from argtyped import Arguments, Choices
  
  class BaseArgs(Arguments):
      foo: int = 0
  
  class DerivedArgs(BaseArgs):
      foo: Choices["a", "b"]
  ```
- `Optional[Literal[...]]` is now correctly supported.
- `Optional[Switch]` is now correctly detected (although invalid).

## [0.2.0] - 2020-06-15
### Added
- Literal types: `Literal`. They act mostly the same as `Choices`.

### Changed
- `Arguments` is now pickle-able.

## [0.1] - 2020-02-16
### Added
- The base of everything: the `Argument` class.
- Choice types: `Choices`.
- Custom enum types: `Enum`. They differ from built-in `Enum` in that `auto()` produces the enum name instead of
  numbers.
- Switch types: `Switch`.

[Unreleased]: https://github.com/huzecong/argtyped/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/huzecong/argtyped/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/huzecong/argtyped/compare/v0.1...v0.2.0
[0.1]: https://github.com/huzecong/argtyped/releases/tag/v0.1
