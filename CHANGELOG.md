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

[Unreleased]: https://github.com/dlaidig/vqf/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/dlaidig/vqf/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/dlaidig/vqf/releases/tag/v1.0.0
