"""Real Python-side check of the `det_ind` feature (PR #160 / issue #156).

Drives the native trackforge module from Python and verifies:
  1. Every returned track is now a 5-tuple (track_id, tlwh, score, class_id, det_ind).
  2. `det_ind` is an int (or None) and maps back into the same frame's detection list.
  3. Building a per-track gallery via `detections[track.det_ind]` yields stable,
     identity-consistent embeddings across frames (the Re-ID use case).

Runs on pure stdlib only (no numpy). Usage:
    python3 -m venv /tmp/tf_venv
    /tmp/tf_venv/bin/pip install target/wheels/trackforge-*.whl
    /tmp/tf_venv/bin/python tests/python_det_ind_check.py
"""

import math

import trackforge


def norm(v):
    mag = math.sqrt(sum(x * x for x in v))
    return [x / mag for x in v] if mag > 1e-6 else list(v)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


FEAT_A = norm([1.0, 0.0, 0.0, 0.0])  # identity A (dim 4)
FEAT_B = norm([0.0, 1.0, 0.0, 0.0])  # identity B


def check(tracker_cls, name, **kwargs):
    tracker = tracker_cls(**kwargs)
    gallery = {}  # track_id -> list of embeddings recovered via det_ind
    ids_for_a, ids_for_b = set(), set()

    for f in range(12):
        dets = [
            ([100.0 + 5.0 * f, 100.0, 40.0, 80.0], 0.9, 0),
            ([300.0 - 3.0 * f, 120.0, 40.0, 80.0], 0.9, 1),
        ]
        embs = [FEAT_A, FEAT_B]
        tracks = tracker.update(dets)
        for t in tracks:
            assert isinstance(t, tuple) and len(t) == 5, f"expected 5-tuple, got {t!r}"
            tid, tlwh, score, cls, det_ind = t
            assert det_ind is None or isinstance(det_ind, int), f"det_ind={det_ind!r}"
            assert det_ind is not None, "activated track should carry det_ind"
            assert 0 <= det_ind < len(dets), f"det_ind {det_ind} out of range"
            src_box, _, _ = dets[det_ind]
            src_emb = embs[det_ind]
            for k in range(4):
                assert abs(src_box[k] - tlwh[k]) < 1.5, f"box mismatch {src_box} vs {tlwh}"
            # Embedding must belong to one consistent identity.
            (ids_for_b if dot(src_emb, FEAT_B) > dot(src_emb, FEAT_A) else ids_for_a).add(tid)
            gallery.setdefault(tid, []).append(tuple(src_emb))

    assert ids_for_a and ids_for_b, f"{name}: one identity never tracked"
    assert len(ids_for_a) == 1, f"{name}: identity A split across {sorted(ids_for_a)}"
    assert len(ids_for_b) == 1, f"{name}: identity B split across {sorted(ids_for_b)}"
    assert ids_for_a != ids_for_b, f"{name}: ids shared between identities"
    print(f"[OK] {name}: det_ind present & consistent; "
          f"A ids={sorted(ids_for_a)} B ids={sorted(ids_for_b)} gallery ids={sorted(gallery)}")


if __name__ == "__main__":
    check(trackforge.BYTETRACK, "BYTETRACK")
    check(trackforge.BOTSORT, "BOTSORT", appearance_thresh=0.5)
    check(trackforge.SORT, "SORT", min_hits=1)
    check(trackforge.OCSORT, "OCSORT", min_hits=1)
    print("ALL PYTHON det_ind CHECKS PASSED")
