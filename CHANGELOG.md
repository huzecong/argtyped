# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
### Changed

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

[Unreleased]: https://github.com/huzecong/argtyped/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/huzecong/argtyped/compare/v0.1...v0.2.0
[0.1]: https://github.com/huzecong/argtyped/releases/tag/v0.1
