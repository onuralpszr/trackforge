# Adding a tracker

This guide describes the end-to-end process for adding a new multi-object tracker to
trackforge. Every tracker ships for both Rust and Python, is tested in both, is
documented, and credits the paper and original authors it is based on.

Open an issue first (use the "New tracker" issue template) so the algorithm, scope, and
parameters can be agreed before you write code.

## 1. Reuse the shared building blocks

Most of the work is the association logic specific to the algorithm. The motion model,
geometry, and assignment are already implemented and must be reused, not re-implemented:

- `utils::kalman` - the shared 8-dimensional Kalman filter
  (state `[x, y, a, h, vx, vy, va, vh]`).
- `utils::geometry` - `tlwh_to_xyah`, `xyah_to_tlwh`, `iou`, `iou_batch`, `iou_cost_matrix`.
- `utils::assignment` - `greedy_match` and the `iou_match` helper.
- `trackers::common` - the shared track base (Kalman state plus lifecycle counters) and the
  shared `TrackState` enum. Compose these instead of declaring your own.
- For appearance-based trackers, reuse
  `trackers::deepsort::nn_matching::NearestNeighborDistanceMetric` for the feature gallery.

## 2. Module layout

Create `src/trackers/<tracker_name>/` with the same layout every tracker uses:

```text
src/trackers/<tracker_name>/
  mod.rs        public tracker struct, re-exports, crate documentation
  tracker.rs    core association and update loop
  track.rs      per-track state (composes trackers::common base)
  python.rs     PyO3 binding
  README.md     algorithm summary, parameters, credit and citation
```

Register the module in `src/trackers/mod.rs` and the Python class in the `#[pymodule]`
in `src/lib.rs`.

## 3. Public API conventions

- Rust constructor takes plain parameters in a documented order, for example
  `Sort::new(max_age, min_hits, iou_threshold)`.
- Implement the `Tracker` trait so the tracker has the same `update` contract as the others.
  Appearance-based trackers that need the frame keep the image-plus-detections variant.
- The Python class name is the upper-case algorithm name (`SORT`, `BYTETRACK`, `OCSORT`,
  `DEEPSORT`, `DEEPOCSORT`). Python `update` returns
  `List[Tuple[int, List[float], float, int]]` = `(track_id, tlwh, score, class_id)`.
- Appearance-based Python bindings take embeddings directly (the detector-side feature
  extraction lives in the example, not in the binding).

## 4. Tests

- Add Rust unit tests in the tracker module (construction, empty update, confirmation
  after `min_hits`, re-association across frames).
- Add a block to `tests/test_python_bindings.py` exercising construction, empty update,
  the returned tuple shape, and any input validation.
- `cargo test`, `cargo test --features python`, and `cargo clippy --all-targets -- -D warnings`
  must pass.

## 5. Documentation and examples

- Fill in `src/trackers/<tracker_name>/README.md` from
  [`.github/TRACKER_TEMPLATE.md`](../.github/TRACKER_TEMPLATE.md).
- Add a crate-level doc section in `src/lib.rs` (tracker table row plus a runnable example).
- Add a guide page `book/src/trackers/<tracker_name>.md` and link it from
  `book/src/trackers/index.md` and `book/src/SUMMARY.md`.
- Add a Python example `examples/python/<tracker_name>_demo.py` built on
  `examples/python/common.py`, and update `examples/python/README.md` and the README tables.
- Update `docs/roadmap.md` (move the tracker from Planned to Available) and the README
  "Supported Trackers" table.

## 6. Credit and citation

Every tracker must credit its source:

- Link the paper (arXiv id) in the README tracker table and the tracker docs.
- Include a paper header, a one-paragraph summary, a link to the original reference
  implementation, and a BibTeX block in the tracker README and book page.
- Note clearly when the code is a clean-room reimplementation of the published algorithm.

## 7. Commit and PR rules

Follow [`COMMIT_GUIDELINES.md`](COMMIT_GUIDELINES.md). Commit with `git commit -sm`, use a
tracker scope (for example `feat(deep_ocsort): ...`), and keep the PR focused.
