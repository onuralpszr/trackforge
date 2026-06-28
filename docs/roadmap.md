# Roadmap

## Available now

- [SORT](https://arxiv.org/abs/1602.00763)
- [ByteTrack](https://arxiv.org/abs/2110.06864)
- [OC-SORT](https://arxiv.org/abs/2203.14360)
- [DeepSORT](https://arxiv.org/abs/1703.07402)
- [Deep OC-SORT](https://arxiv.org/abs/2302.11813)
- Python bindings and PyPI package
- Rust and Python examples

Every tracker ships for both Python and Rust and is tested in both.

## Planned trackers

These build on the shared `utils::kalman`, `utils::geometry`, and `utils::assignment` cores, so
most of the work is the association logic specific to each method.

| Tracker      | Builds on                               | Paper                                          | Reference                                                     |
| ------------ | --------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------- |
| BoT-SORT     | ByteTrack + Re-ID + camera motion       | [2206.14651](https://arxiv.org/abs/2206.14651) | [NirAharon/BoT-SORT](https://github.com/NirAharon/BoT-SORT)   |
| StrongSORT   | DeepSORT + stronger Re-ID               | [2202.13514](https://arxiv.org/abs/2202.13514) | [dyhBUPT/StrongSORT](https://github.com/dyhBUPT/StrongSORT)   |
| StrongSORT++ | StrongSORT + camera motion (AFLink/GSI) | [2202.13514](https://arxiv.org/abs/2202.13514) | [dyhBUPT/StrongSORT](https://github.com/dyhBUPT/StrongSORT)   |
| TrackTrack   | Track-centric online association        | [CVPR 2025](https://arxiv.org/abs/2504.20083)  | [kamkyu94/TrackTrack](https://github.com/kamkyu94/TrackTrack) |
| FastTracker  | Lightweight real-time association       | [2507.06310](https://arxiv.org/abs/2507.06310) | upstream reference                                            |
| Norfair      | Distance-based, detector-agnostic       | -                                              | [tryolabs/norfair](https://github.com/tryolabs/norfair)       |

## Exploring

- Joint detection and tracking ([FairMOT](https://github.com/ifzhang/FairMOT), [CenterTrack](https://github.com/xingyizhou/CenterTrack))
- Transformer-based trackers ([TrackFormer](https://github.com/timmeinhardt/trackformer), [MOTR](https://github.com/megvii-research/MOTR))

Have a request or want to help land one of these? Open an issue or a PR.
