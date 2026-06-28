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


def test_ocsort_round2_rematch_fast_moving():
    # Round-2 re-matching fires when the Kalman-predicted position has drifted far
    # from the detection but the last *observed* position still overlaps it.
    # Create a fast-moving track so the prediction overshoots significantly.
    t = trackforge.OCSORT(
        max_age=5, min_hits=1, iou_threshold=0.3, delta_t=3, inertia=0.2
    )
    # Frame 1: establish track at x=0
    t.update([([0.0, 0.0, 50.0, 100.0], 0.9, 0)])
    # Frame 2: large jump to x=200 — Kalman learns a strong rightward velocity
    t.update([([200.0, 0.0, 50.0, 100.0], 0.9, 0)])
    # Frame 3: detection reappears at x=200 (last observed), not at Kalman prediction ~x=400.
    # Round-1 IoU(predicted≈[400,0,50,100], det=[200,0,50,100]) = 0 → unmatched.
    # Round-2 IoU(last_obs=[200,0,50,100], det=[200,0,50,100]) = 1.0 → matched.
    tracks = t.update([([200.0, 0.0, 50.0, 100.0], 0.9, 0)])
    assert len(tracks) >= 1


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
    t.update(det, emb)
    tracks = t.update(det, emb)
    assert len(tracks) == 1


# ---------------------------------------------------------------------------
# BYTETRACK
# ---------------------------------------------------------------------------


def test_bytetrack_constructor():
    t = trackforge.BYTETRACK()
    assert t is not None


def test_bytetrack_update_empty():
    t = trackforge.BYTETRACK()
    assert t.update([]) == []


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


def test_sort_update_empty():
    t = trackforge.SORT()
    assert t.update([]) == []


def test_sort_update_confirmed_after_min_hits():
    t = trackforge.SORT(max_age=1, min_hits=3, iou_threshold=0.3)
    det = [([100.0, 100.0, 50.0, 100.0], 0.9, 0)]
    for _ in range(3):
        t.update(det)
    tracks = t.update(det)
    assert len(tracks) == 1


# ---------------------------------------------------------------------------
# DEEPOCSORT
# ---------------------------------------------------------------------------


def test_deepocsort_constructor():
    t = trackforge.DEEPOCSORT(
        max_age=30,
        min_hits=1,
        iou_threshold=0.3,
        delta_t=3,
        inertia=0.2,
        appearance_weight=0.5,
        max_cosine_distance=0.2,
        nn_budget=100,
    )
    assert t is not None


def test_deepocsort_update_empty():
    t = trackforge.DEEPOCSORT(min_hits=1)
    assert t.update([]) == []


def test_deepocsort_motion_only():
    # No embeddings: tracks on motion alone (pure OC-SORT behaviour).
    t = trackforge.DEEPOCSORT(min_hits=1)
    tracks = t.update([([100.0, 100.0, 50.0, 100.0], 0.9, 0)])
    assert len(tracks) == 1
    track_id, tlwh, score, class_id = tracks[0]
    assert track_id == 1
    assert len(tlwh) == 4


def test_deepocsort_with_embeddings_keeps_id():
    t = trackforge.DEEPOCSORT(min_hits=1, max_cosine_distance=0.3)
    emb = [[1.0, 0.0, 0.0]]
    first = t.update([([100.0, 100.0, 50.0, 100.0], 0.9, 0)], emb)
    track_id = first[0][0]
    second = t.update([([104.0, 100.0, 50.0, 100.0], 0.9, 0)], emb)
    assert len(second) == 1
    assert second[0][0] == track_id


def test_deepocsort_mismatched_embeddings_raises():
    t = trackforge.DEEPOCSORT(min_hits=1)
    with pytest.raises(ValueError):
        t.update([([100.0, 100.0, 50.0, 100.0], 0.9, 0)], [[1.0], [2.0]])


def test_deepocsort_camera_motion_keeps_id():
    t = trackforge.DEEPOCSORT(min_hits=1)
    first = t.update([([100.0, 100.0, 50.0, 100.0], 0.9, 0)])
    track_id = first[0][0]
    # Camera pans right by 200px; the object now appears at x=300. The affine
    # [a, b, tx, c, d, ty] warps the prediction so the track is kept.
    camera_motion = [1.0, 0.0, 200.0, 0.0, 1.0, 0.0]
    second = t.update([([300.0, 100.0, 50.0, 100.0], 0.9, 0)], [], camera_motion)
    assert len(second) == 1
    assert second[0][0] == track_id
