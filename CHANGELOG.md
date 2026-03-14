# Changelog

All notable changes to this project will be documented in this file.

## [0.1.9] - 2026-03-14

### 📚 Documentation

- 📝 update ByteTrack status to completed by @onuralpszr
- 📝 Update trackforge version in README.md by @onuralpszr
- 📝 add DeepSORT to Trackforge description by @onuralpszr

### Release

- 📦 bump version to 0.1.9
## [0.1.8] - 2026-01-07

### 🐛 Bug Fixes

- *(docs)* 🐛 fix logo paths in README.md for light/dark themes by @onuralpszr
## [0.1.7] - 2026-01-07

### 🚀 Features

- *(tracker)* ✨ add Deep SORT tracker implementation- Implement NearestNeighborDistanceMetric for cosine/euclidean matching- Add cascade matching with appearance and IoU distance- Implement track lifecycle (tentative, confirmed, deleted states)- Add Kalman filter extensions for Deep SORT state management- Include comprehensive documentation and examples by @onuralpszr
- *(python)* 🐍 add Python bindings for Deep SORT tracker by @onuralpszr
- ✨ add Deep SORT tracker with Python bindings #21 by @onuralpszr

### 🐛 Bug Fixes

- *(clippy)* 🔧 fix clippy lint errors by @onuralpszr
- *(clippy)* 🔧 fix remaining clippy lint errors by @onuralpszr
- *(ci)* 🔧 remove opencv/ort from dev-dependencies by @onuralpszr
- *(ci)* 🔧 gate advanced example dependencies behind feature flag by @onuralpszr

### 🚜 Refactor

- *(python)* 📁 move type stubs to python/ directory by @onuralpszr

### 📚 Documentation

- *(examples)* 📝 add Deep SORT Python demo with YOLO by @onuralpszr
- 📝 update documentation for Deep SORT tracker by @onuralpszr
- *(examples)* 📝 add Rust Deep SORT examples by @onuralpszr
- 📝 add better TODO section for trackers by @onuralpszr
- 📝 add msrv badge to readme by @onuralpszr

### 🧪 Testing

- ✅ add comprehensive unit tests for Deep SORT by @onuralpszr
- ✅ add more tracker tests for improved coverage by @onuralpszr
## [0.1.6] - 2025-12-31

### 🚀 Features

- ✨ add SORT tracker implementation with Python bindings by @onuralpszr

### 📚 Documentation

- 📝 add Python tracking examples with YOLO and RT-DETR by @onuralpszr
- 📝 update roadmap to mark SORT as completed by @onuralpszr
## [0.1.5] - 2025-12-30

### 🚀 Features

- ✨ add new asset project logos for dark & light themes  by @onuralpszr
- ✨ add initial documentation and deployment workflow for Trackforge  by @onuralpszr

### 🐛 Bug Fixes

- *(docs)* 🐞 update logo path and size to show proper logo by @onuralpszr

### ⚙️ Miscellaneous Tasks

- 👷 change doc action use uv and check in PRs #19 by @onuralpszr
- 📝 update changelog for v0.1.5 release by @onuralpszr
## [0.1.4] - 2025-12-26

### ⚙️ Miscellaneous Tasks

- 📦 bump to 0.1.4 with fixed metadata for publishing by @onuralpszr
## [0.1.3] - 2025-12-26

### 🚀 Features

- ✨ Add initial project structure with CI configuration and Python bindings by @onuralpszr
- ✨ Implement initial structure for trackers and types, add Python bindings by @onuralpszr
- ✨ Add AppearanceExtractor trait and DeepSort tracker implementation by @onuralpszr
- ✨ Add .editorconfig for consistent coding styles across files by @onuralpszr
- ✨ Add initial Codecov configuration for coverage reporting by @onuralpszr
- ✨ Add contribution guidelines to enhance collaboration and quality standards by @onuralpszr
- ✨ Add initial Commitizen configuration for standardized commit messages by @onuralpszr
- ✨ Add alias for xtask to streamline package execution by @onuralpszr
- ✨ Add CODEOWNERS file to define repository maintainers by @onuralpszr
- ✨ Add security audit workflow for Cargo dependencies by @onuralpszr
- ✨ Update actions/checkout to version 6 in security audit workflow by @onuralpszr
- ✨ Add Dependabot configuration for automated dependency updates  by @onuralpszr
- ✨ Update .gitignore to include additional file types for weights and media by @onuralpszr
- ✨ Update dependencies and add example for byte tracking by @onuralpszr
- ✨ Enhance README with detailed usage examples and installation instructions by @onuralpszr
- ✨ Add Python and Rust examples for ByteTrack tracking functionality by @onuralpszr
- ✨ Implement ByteTrack tracker and integrate with Python bindings by @onuralpszr
- ✨ Update .gitignore to include mypycache files by @onuralpszr
- ✨ Update dependencies and clean up unused code in Cargo.toml and mod.rs by @onuralpszr
- ✨ Add ignore rule for specific RustSec advisory in security audit workflow by @onuralpszr
- ✨ Add audit configuration file for Cargo security auditing by @onuralpszr
- ✨ Add initial configuration for cargo-deny to manage advisories, licenses, bans, and sources by @onuralpszr
- ✨ Update .gitignore to include cargo advisory database lock file by @onuralpszr
- ✨ Refactor ByteTrack cost matrix calculation and update KalmanFilter error handling by @onuralpszr
- ✨ Implement ByteTrack tracker and integrate with Python bindings #10 by @onuralpszr
- ✨ Update CI workflow for PyPI and Crates.io publishing; bump version to 0.1.3 and enhance documentation  by @onuralpszr

### 🐛 Bug Fixes

- 🐛 Update pyo3 dependency configuration and adjust maturin features by @onuralpszr
- 🐛 Update artifact upload actions and naming conventions in CI workflow by @onuralpszr
- 🐛 Allow dead code warning for extractor field in DeepSort struct by @onuralpszr
- *(byte_track)* 🐛 Update test assertions to use variable for track ID consistency by @onuralpszr
- *(ci)* 🐞 download artifacts to dist/ to avoid uploading .cargo dir by @onuralpszr

### 📚 Documentation

- ✏️ Add README.md for ByteTrack algorithm documentation by @onuralpszr

### 🧪 Testing

- *(byte_track)* 🧪 Add Tests for STrack and ByteTrack in byte_track module by @onuralpszr

### ⚙️ Miscellaneous Tasks

- 👷 upgrade all of the action versions to make sure CI works by @onuralpszr
- 👷 Add Rust check job to CI workflow by @onuralpszr
- 📦 Update dependencies and configuration files by @onuralpszr
- 👷 add initial codecov github action configuration by @onuralpszr
- 🧹 remove example comments from audit.toml  by @onuralpszr
- 📦 bump version from 0.1.1 to 0.1.2  by @onuralpszr

