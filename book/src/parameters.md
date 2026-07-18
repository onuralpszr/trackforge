# Parameters

The parameters are identical across Python and Rust. In Python they are keyword arguments to the
tracker class. In Rust they are positional arguments to `Tracker::new(...)`, or fields on a params
struct passed to `Tracker::from_params(...)`.

Two settings recur in almost every tracker. How many frames a lost track survives before it is
dropped, called `max_age` in most trackers and `track_buffer` in ByteTrack and BoT-SORT. And how
many matched frames in a row confirm a new track, called `min_hits` in most and `n_init` in
DeepSORT. In Rust these two live in `CommonParams`, which each tracker's params struct embeds as
`common`.

IoU thresholds come in two opposite flavors. A minimum IoU (`iou_threshold`, higher is stricter) in
SORT, OC-SORT, and Deep OC-SORT. And a maximum IoU distance of one minus IoU (`match_thresh`,
`max_iou_distance`, lower is stricter) in ByteTrack, BoT-SORT, and DeepSORT. The full table with a
plain description of every parameter is on the
[documentation site](https://onuralpszr.github.io/trackforge/parameters.html).

Each tracker's chapter documents its own parameters and tuning advice:

- [SORT parameters](./trackers/sort.md#parameters)
- [ByteTrack parameters](./trackers/byte_track.md#parameters)
- [OC-SORT parameters](./trackers/ocsort.md#parameters)
- [DeepSORT parameters](./trackers/deepsort.md#parameters)

The same tables are also published on the
[documentation site](https://onuralpszr.github.io/trackforge/parameters.html).
