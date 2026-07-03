"""End-to-end BoT-SORT tracking scenarios.

These drive the tracker over multiple frames with synthetic detections (no detector)
and assert real tracking behaviour: stable ids, occlusion recovery, camera-motion
compensation, appearance disambiguation through a crossing, and motion-only parity
with ByteTrack.
"""

import trackforge


def box(x, y, w=40, h=80):
    return [float(x), float(y), float(w), float(h)]


def id_near(tracks, near_x):
    """Return the id of the track whose box center-x is closest to ``near_x``."""
    return min(tracks, key=lambda t: abs(t[1][0] - near_x))[0]


def test_two_objects_moving_keep_stable_distinct_ids():
    t = trackforge.BOTSORT()
    seen_a, seen_b = set(), set()
    for f in range(20):
        dets = [(box(100 + f * 8, 100), 0.9, 0), (box(500 - f * 6, 300), 0.85, 0)]
        tracks = t.update(dets)
        assert len(tracks) == 2, f"frame {f}: expected 2 tracks, got {len(tracks)}"
        seen_a.add(id_near(tracks, 100 + f * 8))
        seen_b.add(id_near(tracks, 500 - f * 6))
    assert len(seen_a) == 1, f"object A changed id: {seen_a}"
    assert len(seen_b) == 1, f"object B changed id: {seen_b}"
    assert seen_a != seen_b


def test_occlusion_recovery_keeps_id():
    t = trackforge.BOTSORT(track_buffer=30)
    tid = t.update([(box(200, 200), 0.9, 0)])[0][0]
    for f in range(1, 6):
        assert t.update([]) == [], f"frame {f}: track should be hidden while occluded"
    tracks = t.update([(box(205, 200), 0.9, 0)])
    assert len(tracks) == 1
    assert tracks[0][0] == tid, f"id not recovered: {tracks[0][0]} != {tid}"


def test_camera_pan_keeps_id():
    t = trackforge.BOTSORT()
    tid = t.update([(box(400, 200), 0.9, 0)])[0][0]
    for f in range(1, 8):
        # Camera pans right 30px/frame: the object appears to move right, but the
        # affine [1, 0, 30, 0, 1, 0] warps the prediction so the id holds.
        appear_x = 400 + f * 30
        cmc = [1.0, 0.0, 30.0, 0.0, 1.0, 0.0]
        tracks = t.update([(box(appear_x, 200), 0.9, 0)], [], cmc)
        assert len(tracks) == 1, f"frame {f}: lost the track under pan"
        assert tracks[0][0] == tid, f"frame {f}: id changed under pan ({tracks[0][0]})"


def test_appearance_prevents_swap_through_crossing():
    # Frame 0 assigns id 1 to object A (starts left, emb_a, moves right) and id 2 to
    # object B (starts right, emb_b, moves left). If appearance holds through the
    # crossing, id 1 ends on the right and id 2 on the left; a swap would flip that.
    t = trackforge.BOTSORT(appearance_thresh=0.4)
    emb_a = [1.0, 0.0, 0.0]
    emb_b = [0.0, 1.0, 0.0]
    x1_first = x2_first = None
    x1_last = x2_last = None
    for f in range(12):
        ax, bx = 200 + f * 20, 440 - f * 20
        tracks = t.update(
            [(box(ax, 200), 0.9, 0), (box(bx, 200), 0.9, 1)], [emb_a, emb_b]
        )
        assert len(tracks) == 2, f"frame {f}: expected 2 tracks, got {len(tracks)}"
        by_id = {trk[0]: trk[1][0] for trk in tracks}
        assert set(by_id) == {1, 2}, (
            f"frame {f}: unexpected ids {set(by_id)} (fragmentation)"
        )
        if f == 0:
            x1_first, x2_first = by_id[1], by_id[2]
        x1_last, x2_last = by_id[1], by_id[2]
    assert x1_last > x1_first, (
        f"id 1 did not follow the rightward object: {x1_first}->{x1_last}"
    )
    assert x2_last < x2_first, (
        f"id 2 did not follow the leftward object: {x2_first}->{x2_last}"
    )


def test_motion_only_parity_with_bytetrack():
    # With no embeddings and no camera motion, BoT-SORT tracks like ByteTrack.
    bt = trackforge.BYTETRACK()
    bs = trackforge.BOTSORT()
    for f in range(10):
        dets = [(box(150 + f * 10, 250), 0.9, 0)]
        assert len(bt.update(dets)) == len(bs.update(dets)) == 1, f"frame {f}: mismatch"
