"""Integration tests for the Python bindings — exercises Rust glue code."""

import pytest
import trackforge


# ---------------------------------------------------------------------------
# OCSORT
# ---------------------------------------------------------------------------


def test_ocsort_constructor():
    t = trackforge.OCSORT(
        max_age=30, min_hits=1, iou_threshold=0.3, delta_t=3, inertia=0.2
    )
    assert t is not None


def test_ocsort_update_empty():
    t = trackforge.OCSORT(
        max_age=30, min_hits=1, iou_threshold=0.3, delta_t=3, inertia=0.2
    )
    assert t.update([]) == []


def test_ocsort_update_returns_track():
    t = trackforge.OCSORT(
        max_age=30, min_hits=1, iou_threshold=0.3, delta_t=3, inertia=0.2
    )
    tracks = t.update([([100.0, 100.0, 50.0, 100.0], 0.9, 0)])
    assert len(tracks) == 1
    track_id, tlwh, score, class_id = tracks[0]
    assert track_id == 1
    assert len(tlwh) == 4
    assert class_id == 0


def test_ocsort_confirmed_after_min_hits():
    t = trackforge.OCSORT(
        max_age=30, min_hits=3, iou_threshold=0.3, delta_t=3, inertia=0.2
    )
    det = [([100.0, 100.0, 50.0, 100.0], 0.9, 0)]
    for _ in range(3):
        t.update(det)
    tracks = t.update(det)
    assert len(tracks) == 1


# ---------------------------------------------------------------------------
# DEEPSORT
# ---------------------------------------------------------------------------


def test_deepsort_constructor():
    t = trackforge.DEEPSORT(
        max_age=70,
        n_init=3,
        max_iou_distance=0.7,
        max_cosine_distance=0.2,
        nn_budget=100,
    )
    assert t is not None


def test_deepsort_update_empty():
    t = trackforge.DEEPSORT()
    assert t.update([], []) == []


def test_deepsort_mismatched_lengths_raises():
    t = trackforge.DEEPSORT()
    with pytest.raises(ValueError):
        t.update([([100.0, 100.0, 50.0, 100.0], 0.9, 0)], [])


def test_deepsort_update_with_embedding():
    t = trackforge.DEEPSORT(
        max_age=70,
        n_init=1,
        max_iou_distance=0.7,
        max_cosine_distance=0.2,
        nn_budget=100,
    )
    det = [([100.0, 100.0, 50.0, 100.0], 0.9, 0)]
    emb = [[0.1] * 128]
    tracks = t.update(det, emb)
    assert len(tracks) == 1


# ---------------------------------------------------------------------------
# BYTETRACK
# ---------------------------------------------------------------------------


def test_bytetrack_constructor():
    t = trackforge.BYTETRACK()
    assert t is not None


def test_bytetrack_update_returns_track():
    t = trackforge.BYTETRACK(
        track_thresh=0.5, track_buffer=30, match_thresh=0.8, det_thresh=0.6
    )
    tracks = t.update([([100.0, 100.0, 50.0, 100.0], 0.9, 0)])
    assert len(tracks) == 1


# ---------------------------------------------------------------------------
# SORT
# ---------------------------------------------------------------------------


def test_sort_constructor():
    t = trackforge.SORT()
    assert t is not None


def test_sort_update_confirmed_after_min_hits():
    t = trackforge.SORT(max_age=1, min_hits=3, iou_threshold=0.3)
    det = [([100.0, 100.0, 50.0, 100.0], 0.9, 0)]
    for _ in range(3):
        t.update(det)
    tracks = t.update(det)
    assert len(tracks) == 1
