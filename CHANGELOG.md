<!--
SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>

SPDX-License-Identifier: MIT

Sections:
### Added (for new features)
### Changed (for changes in existing functionality)
### Deprecated (for soon-to-be removed features)
### Removed (for now removed features)
### Fixed (for any bug fixes)
### Security (in case of vulnerabilities)
-->
# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.1.0] - 2026-01-29
### Added
- Python type hints (for both the wrappers and the pure Python implementations).
### Changed
- Change minimum required Python from 3.7 to 3.10 (as Python <= 3.9 is end-of-life).
- Improved CMake support (contributed by [SanyaNya](https://github.com/SanyaNya), see #36).
### Fixed
- Add passthrough fallback in Butterwoth filter implementation when cutoff frequency becomes close to Nyquist frequency (see #29).
- Various adjustments in the C++ implementation to increase compatibility and fix compile warnings.
    - Fixed compile issues due to min/max macros on Windows (contributed by [SanyaNya](https://github.com/SanyaNya), see #35).
    - Fixed `-Wdouble-promotion` and `-Warray-parameter` compiler warnings (see #32).
    - Use `cmath` instead of `math.h`, and provide own constants instead of `M_PI` and `M_SQRT2` (see #32).

## [2.0.1] - 2024-11-15
### Fixed
- Removed obsolete (and unused) `restMagLpB` and `restMagLpA` filter coefficient calculation in Matlab implementation.
- Added `lastAccCorrAngularRate` member to `PyVQFState`.

## [2.0.0] - 2022-10-28
### Changed
- All references to the publication are updated to the published journal article.
- Rest detection is now based on gyroscopes and accelerometers only and does not make use of magnetometer measurements.
  (In practice, gyroscopes and accelerometers are more sensitive for rest detection, and the use of magnetometers does
  not add any value. In contrast, miscalibrated magnetometers could previously prevent rest from being detected.)
### Fixed
- Fixed typo `getmagDistDetected` -> `getMagDistDetected` in Matlab implementation.

## [1.0.0] - 2022-04-01
### Added
- Initial release.

[Unreleased]: https://github.com/dlaidig/vqf/compare/v2.1.0...HEAD
[2.1.0]: https://github.com/dlaidig/vqf/compare/v2.0.1...v2.1.0
[2.0.1]: https://github.com/dlaidig/vqf/compare/v2.0.0...v2.0.1
[2.0.0]: https://github.com/dlaidig/vqf/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/dlaidig/vqf/releases/tag/v1.0.0
