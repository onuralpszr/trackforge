<!--
Fill-in-the-blanks spec for a new tracker. Copy this file to
src/trackers/<tracker_name>/README.md and replace every <...> placeholder.
Remove this comment and any sections that do not apply.
See docs/adding-a-tracker.md for the full process.
-->

# <Tracker Name>: <Full Algorithm Name>

This module implements the <Tracker Name> algorithm.

> **<Paper Title>**
> <Author list>
> <Venue and year>
> [arXiv:<id>](https://arxiv.org/abs/<id>)

## Algorithm overview

<One or two paragraphs describing how the algorithm associates detections with tracks.
Name the key mechanisms (for example motion model, appearance, two-stage matching).>

## Builds on

- `utils::kalman` - <how the Kalman filter is used>
- `utils::geometry` - <which IoU or conversion helpers>
- `utils::assignment` - <greedy_match / iou_match>
- `trackers::common` - shared track base and `TrackState`
- <other reused modules, for example deepsort::nn_matching for appearance>

## Parameters

| Parameter | Default     | Description        |
| --------- | ----------- | ------------------ |
| `<param>` | `<default>` | <what it controls> |

**Tuning:** <short guidance on which parameters to raise or lower for which scenarios>

## Rust API

```rust
use trackforge::trackers::<tracker_name>::<TrackerType>;

let mut tracker = <TrackerType>::new(<args>);
let tracks = tracker.update(vec![([100.0, 100.0, 50.0, 100.0], 0.9, 0)]);
for t in tracks {
    println!("ID: {}, Box: {:?}", t.track_id, t.tlwh);
}
```

## Python API

```python
from trackforge import <PYNAME>

tracker = <PYNAME>(<args>)
detections = [([100.0, 100.0, 50.0, 100.0], 0.9, 0)]
tracks = tracker.update(detections)  # add embeddings for appearance trackers
for track_id, tlwh, score, class_id in tracks:
    print(f"ID: {track_id}, Box: {tlwh}")
```

## Tests

- [ ] Rust unit tests (construction, empty update, confirmation after `min_hits`, re-association)
- [ ] Python block in `tests/test_python_bindings.py`
- [ ] `cargo test`, `cargo test --features python`, `cargo clippy --all-targets -- -D warnings` pass

## Documentation and examples

- [ ] `src/lib.rs` crate-doc tracker table row and example
- [ ] `book/src/trackers/<tracker_name>.md` plus index and SUMMARY links
- [ ] `examples/python/<tracker_name>_demo.py` on `examples/python/common.py`
- [ ] `examples/python/README.md`, README tables, and `docs/roadmap.md` updated

## Credit

Clean-room Rust implementation of the algorithm described in the paper above. Original
reference implementation: [<owner>/<repo>](https://github.com/<owner>/<repo>).

## Citation

```bibtex
@article{<key>,
  title={<Paper Title>},
  author={<Authors>},
  journal={<Venue>},
  year={<year>}
}
```
