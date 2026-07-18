//! Regression guard for the shared ByteTrack and BoT-SORT cascade.
//!
//! ByteTrack and BoT-SORT drive the same two-stage cascade; the only difference is the
//! stage-one cost (plain IoU vs appearance-fused) and BoT-SORT's camera warp. With no
//! embeddings and no camera motion, BoT-SORT reduces to ByteTrack, so both must produce
//! identical output on a motion-only sequence. This test locks that in.

use trackforge::trackers::botsort::BotSort;
use trackforge::trackers::byte_track::ByteTrack;

/// Deterministic sequence: object A steady, object B occluded on frames 4..7 and
/// returning at low score on frame 7, object C entering at frame 2 and leaving after 9.
fn sequence() -> Vec<Vec<([f32; 4], f32, i64)>> {
    let mut frames = Vec::new();
    for f in 0..14 {
        let mut dets = Vec::new();
        dets.push(([10.0 + f as f32 * 6.0, 40.0, 40.0, 90.0], 0.92, 0));
        if !(4..7).contains(&f) {
            let score = if f == 7 { 0.45 } else { 0.88 };
            dets.push(([200.0 + f as f32 * 4.0, 60.0, 45.0, 95.0], score, 0));
        }
        if (2..10).contains(&f) {
            dets.push(([400.0 - f as f32 * 5.0, 120.0, 38.0, 80.0], 0.8, 1));
        }
        frames.push(dets);
    }
    frames
}

fn ids(mut rows: Vec<u64>) -> Vec<u64> {
    rows.sort_unstable();
    rows
}

#[test]
fn bytetrack_and_botsort_agree_on_motion_only() {
    let mut byte = ByteTrack::new(0.5, 30, 0.8, 0.6);
    let mut bot = BotSort::new(0.5, 30, 0.8, 0.6, 0.5, 0.25);

    for (f, dets) in sequence().into_iter().enumerate() {
        let mut b: Vec<_> = byte
            .update(dets.clone())
            .into_iter()
            .map(|t| (t.track_id, t.tlwh, (t.score * 100.0) as i32))
            .collect();
        let mut o: Vec<_> = bot
            .update(dets, &[])
            .into_iter()
            .map(|t| (t.track_id, t.tlwh, (t.score * 100.0) as i32))
            .collect();
        b.sort_by_key(|r| r.0);
        o.sort_by_key(|r| r.0);
        assert_eq!(b, o, "ByteTrack and BoT-SORT diverged on frame {f}");
    }
}

#[test]
fn occluded_object_keeps_its_id_after_the_gap() {
    let mut byte = ByteTrack::new(0.5, 30, 0.8, 0.6);
    let mut per_frame: Vec<Vec<u64>> = Vec::new();
    for dets in sequence() {
        per_frame.push(byte.update(dets).into_iter().map(|t| t.track_id).collect());
    }
    // Frame 0 establishes objects A (id 1) and B (id 2).
    assert_eq!(ids(per_frame[0].clone()), vec![1, 2]);
    // Object B is gone during the occlusion window.
    assert!(!per_frame[5].contains(&2));
    // Object B returns at high score on frame 8 and recovers its original id.
    assert!(
        per_frame[8].contains(&2),
        "id 2 should recover after the gap"
    );
}
