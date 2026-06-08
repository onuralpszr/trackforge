# Changelog

All notable changes to this project will be documented in this file.

## [0.3.0] - 2026-06-08

### 🚜 Refactor

- Centralize tlwh/xyah box conversions in `utils::geometry`
- Share the greedy assignment core in `utils::assignment`
- Share the IoU cost-matrix construction in `utils::geometry`
- Share the DeepSORT tracker construction across Rust and Python
- Use bool vectors instead of HashSet in `greedy_match`

### 📚 Documentation

- Overhaul the README and docs site (unified examples, badges, parameters and roadmap pages)
- Add an mdBook guide, published at `/book` alongside the zensical site
- Fix the docs site URLs so the `.html` links resolve
- Document prek for the git hooks in CONTRIBUTING

### 📦 Dependencies

- Bump nalgebra from 0.34 to 0.35 (raises the MSRV to 1.89)

### ⚙️ Miscellaneous

- Parallelize the Rust CI into separate test, lint, and doc jobs
- Add tests for the extractor error path, the empty IoU guards, and the degenerate IoU case

## [0.2.0] - 2026-05-07

### 🚀 Features

- *(ocsort)* 🚀 add OC-SORT tracker implementation by @onuralpszr
- *(ocsort)* 🚀 add OC-SORT tracker implementation #72 by @onuralpszr
- *(deepsort)* 🚀 enhance TrackState and Track structure with additional states and properties by @onuralpszr

### 🐛 Bug Fixes

- *(deps)* Bump rustls-webpki and tar to patch RUSTSEC-2026-0049/0067/0068/0098/0099; ignore yanked core2 by @onuralpszr
- 🐞 update domain-specific tokens in typos configuration for clarity by @onuralpszr
- 🐞 update clippy hook arguments to improve compatibility with fresh dev environments by @onuralpszr
- 🐞 update bounding box feature vector creation and tracker update calls for consistency by @onuralpszr
- *(deepsort)* Rename _n_init/_max_age fields — they are actively used, not dead code by @onuralpszr
- *(deepsort)* Replace bare .unwrap() in cascade matching with .expect() by @onuralpszr
- 🐞 correct terminology from OUR to ORU in documentation and code comments by @onuralpszr
- 🐞 replace deprecated pyo3::prepare_freethreaded_python() with pyo3::Python::initialize() in tests by @onuralpszr
- 🐞 create virtual environment for installing maturin and pytest in Codecov workflow by @onuralpszr
- *(ocsort)* Replace bare unwrap() with expect() for observations.last() by @onuralpszr
- *(dependencies)* Update ort version to 2.0.0-rc.12 by @onuralpszr
- *(dependencies)* 🐛 downgrade ort dependency version to 2.0.0-rc.10 for usls by @onuralpszr

### 🚜 Refactor

- *(python)* ♻️ rename tracker exports to all-caps (BYTETRACK, SORT, OCSORT, DEEPSORT) by @onuralpszr
- 🧹 remove unused Python test modules for DeepSort and OC-SORT by @onuralpszr
- *(deepsort)* Fix clippy warning and optimize partial_fit lookup by @onuralpszr
- *(bytetrack)* Eliminate per-call KF allocations and dead code by @onuralpszr
- *(deepsort)* Use total_cmp for NaN-safe cost sorting by @onuralpszr
- *(sort)* ♻️ replace unwrap with total_cmp for cost sorting by @onuralpszr
- *(kalman)* ♻️ update comments and enhance measurement vector definitions by @onuralpszr
- *(tests)* 🧪 improve comments for re_activate lost track test case by @onuralpszr
- ♻️ code clean ups and documentation fixes and cargo fix #73 by @onuralpszr

### 📚 Documentation

- 📝 pypi download badge added by @onuralpszr
- 📝 improve readme roadmap and example section by @onuralpszr
- 📝 improve readme and fix typo and remove unused deps by @onuralpszr
- 📝 correct code block syntax for architecture section in README by @onuralpszr
- 📝 update return type documentation for update method in PyDeepSort by @onuralpszr
- 📝 update architecture diagram for clarity and improve formatting in README by @onuralpszr
- Small clean up in readme by @onuralpszr
- 📝 Add TrackTrack to the tracking methods list by @onuralpszr
- 📝 Add dependency status badge to README by @onuralpszr
- 📝 Add dependency status badge to README #67 by @onuralpszr
- *(index)* Rewrite mkdocs index with full per-tracker reference by @onuralpszr
- *(python)* Replace stub with complete Python API reference by @onuralpszr
- *(lib)* Expand crate-level Rust documentation by @onuralpszr
- *(examples)* Replace broken mkdocs file-include syntax with inline code by @onuralpszr
- *(lib)* Expand crate-level Rust documentation with badges and full examples by @onuralpszr
- 📝 documentation fixes and small bug fixes #71 by @onuralpszr
- 📝 update all Python class names to all-caps across docs and README by @onuralpszr
- *(types)* 📝 improve BoundingBox documentation and add new fields by @onuralpszr
- 📝 add missing MSRV badge to README by @onuralpszr

### 🎨 Styling

- Apply prettier to markdown and yaml files by @onuralpszr

### 🧪 Testing

- 🧪 add instance isolation and sequential ID tests for ByteTrack, DeepSort, and SORT trackers by @onuralpszr
- 🧪 add unit tests for PyDeepSort and PyOcSort functionality by @onuralpszr
- 🧪 add integration tests for Python bindings of OCSORT, DEEPSORT, BYTETRACK, and SORT by @onuralpszr
- 🐞 fixing embedding test in deepsort by @onuralpszr
- 🧪 add empty update tests for BYTETRACK and SORT by @onuralpszr
- 🧪 add OCSORT round2 rematch after gap test by @onuralpszr
- 🧪 update OCSORT round2 rematch test for fast-moving tracks by @onuralpszr
- *(coverage)* Add tests for re_activate and observations.last() paths by @onuralpszr

### ⚙️ Miscellaneous Tasks

- 📝 update changelog for version 0.1.9 release by @onuralpszr
- 🧹 formatting fix by @onuralpszr
- 👷 update security audit workflow to include pull request handling and permissions by @onuralpszr
- Remove obsolete commit_plan and reformat_markdown scripts by @onuralpszr
- 👷 update CI workflow to exclude advanced_examples from doc build due to dependency requirements by @onuralpszr
- 👷 update CI workflow by removing pre-commit checks and simplifying cargo steps by @onuralpszr
- Update GitHub Actions to use specific versions for dependencies and tools by @onuralpszr
- *(stubs)* 🧹 update trackforge.pyi for all-caps class names and unified return types by @onuralpszr
- *(examples)* ♻️ update Python examples to all-caps tracker names by @onuralpszr
- *(package)* 📦 update uv.lock by @onuralpszr
- 📦 update trackforge version to 0.2.0 in Cargo files by @onuralpszr
- 📦 update dependencies in Cargo.lock to latest versions by @onuralpszr
- 📦 update rust-version to 1.88 in Cargo.toml by @onuralpszr
- 📦 update taiki-e/install-action to v2.75.30 in CI workflows by @onuralpszr
- *(cargo)* Add include field to restrict crates.io upload by @onuralpszr
- *(workflows)* 👷 update taiki-e/install-action to v2.77.1 in CI, autofix, codencov, and security-audit by @onuralpszr

## [0.1.9] - 2026-03-14

### 📚 Documentation

- 📝 update ByteTrack status to completed by @onuralpszr
- 📝 Update trackforge version in README.md by @onuralpszr
- 📝 add DeepSORT to Trackforge description by @onuralpszr

### ⚙️ Miscellaneous Tasks

- *(changelog)* Update changelog for version 0.1.9 release
- *(audit)* Remove target configuration section from audit.toml
- *(Cargo.toml)* 📦 update rust-version to 1.87
- 🧹 cargo advisory-db added to .gitignore
- *(Cargo.toml)* 📦 update opencv to 0.98.1 and usls to 0.1.11
- *(Cargo.lock)* 📦 update cargo lock file

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

