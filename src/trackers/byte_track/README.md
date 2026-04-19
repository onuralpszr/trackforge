# ByteTrack

> [**ByteTrack: Multi-Object Tracking by Associating Every Detection Box**](https://arxiv.org/abs/2110.06864)
>
> Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan, Ping Luo, Wenyu Liu, Xinggang Wang
>
> _[arXiv 2110.06864](https://arxiv.org/abs/2110.06864)_

**ByteTrack** is a simple, fast and strong multi-object tracker.

## Abstract

Multi-object tracking (MOT) aims at estimating bounding boxes and identities of objects in videos. Most methods obtain identities by associating detection boxes whose scores are higher than a threshold. The objects with low detection scores, e.g. occluded objects, are simply thrown away, which brings non-negligible true object missing and fragmented trajectories. To solve this problem, ByteTrack presents a simple, effective and generic association method, tracking by associating every detection box instead of only the high score ones. For the low score detection boxes, it utilizes their similarities with tracklets to recover true objects and filter out the background detections.

## Original Repository

This is a clean-room Rust implementation of the ByteTrack algorithm as described in the original paper. The official reference implementation can be found at [ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack).

## Citation

```bibtex
@article{zhang2022bytetrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Weng, Fucheng and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2022}
}
```
