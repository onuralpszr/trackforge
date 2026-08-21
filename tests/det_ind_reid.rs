//! Real end-to-end tests for the `det_ind` track-to-detection mapping.
//!
//! These simulate a multi-frame Re-ID pipeline: a moving scene with two visually
//! distinct identities is fed to BoT-SORT with one appearance embedding per
//! detection. For every output track we then recover its source embedding through
//! `detections[track.det_ind]` and check the recovered embedding is consistent
//! with the track's identity. This is exactly the use case from the issue:
//! a long-term tracking / Re-ID database indexed by detection.

use trackforge::trackers::botsort::BotSort;

const DIM: usize = 8;

/// L2-normalise a feature vector (Re-ID embeddings are compared on the unit sphere).
fn norm(v: &[f32]) -> Vec<f32> {
    let mag = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag > 1e-6 {
        v.iter().map(|x| x / mag).collect()
    } else {
        v.to_vec()
    }
}

/// Deterministic identity embeddings: object A and object B are orthogonal, so
/// cosine similarity cleanly separates them.
fn feat_a() -> Vec<f32> {
    let mut f = vec![0.0; DIM];
    f[0] = 1.0;
    norm(&f)
}
fn feat_b() -> Vec<f32> {
    let mut f = vec![0.0; DIM];
    f[1] = 1.0;
    norm(&f)
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Which identity a recovered embedding belongs to (`true` = A, `false` = B).
fn is_a(emb: &[f32]) -> bool {
    cos(emb, &feat_a()) > cos(emb, &feat_b())
}

#[test]
fn det_ind_gallery_is_consistent_across_frames() {
    let mut tracker = BotSort::new(0.5, 30, 0.8, 0.6, 0.5, 0.25);

    // Identity -> track ids that ever mapped to it via det_ind.
    let mut ids_for_a: Vec<u64> = Vec::new();
    let mut ids_for_b: Vec<u64> = Vec::new();

    for f in 0..=20 {
        // Two well-separated moving objects.
        let a_tlwh = [100.0 + 5.0 * f as f32, 100.0 + 0.5 * f as f32, 40.0, 80.0];
        let b_tlwh = [300.0 - 3.0 * f as f32, 120.0, 40.0, 80.0];
        let detections = vec![(a_tlwh, 0.9, 0), (b_tlwh, 0.9, 1)];
        let embeddings = vec![feat_a(), feat_b()];

        let tracks = tracker.update(detections.clone(), &embeddings);

        for t in &tracks {
            let di = t.det_ind.expect("activated track must carry det_ind");
            assert!(di < detections.len(), "det_ind {di} out of range");
            let recovered = &embeddings[di];
            // Every track's recovered embedding must belong to one consistent identity.
            if is_a(recovered) {
                ids_for_a.push(t.track_id);
            } else {
                ids_for_b.push(t.track_id);
            }
            // The recovered box must agree with the detection it claims to be.
            let det_tlwh = &detections[di].0;
            for k in 0..4 {
                assert!(
                    (det_tlwh[k] - t.tlwh[k]).abs() < 1.5,
                    "track {:?} det_ind {di} box mismatch: {:?} vs {:?}",
                    t.tlwh,
                    det_tlwh,
                    t.tlwh
                );
            }
        }
    }

    // Both objects were tracked, so each identity saw at least one track.
    assert!(!ids_for_a.is_empty(), "object A was never tracked");
    assert!(!ids_for_b.is_empty(), "object B was never tracked");

    // Stability: a single identity is never captured under multiple track ids
    // (i.e. det_ind never points a track at the wrong object), and the two
    // identities never share a track id.
    ids_for_a.sort();
    ids_for_b.sort();
    ids_for_a.dedup();
    ids_for_b.dedup();
    assert_eq!(
        ids_for_a.len(),
        1,
        "identity A mapped to multiple track ids: {ids_for_a:?}"
    );
    assert_eq!(
        ids_for_b.len(),
        1,
        "identity B mapped to multiple track ids: {ids_for_b:?}"
    );
    assert_ne!(
        ids_for_a[0], ids_for_b[0],
        "identities A and B shared a track id"
    );
}

#[test]
fn det_ind_survives_reacquisition_after_occlusion() {
    let mut tracker = BotSort::new(0.5, 30, 0.8, 0.6, 0.5, 0.25);

    let hidden = |f: usize| (8..=10).contains(&f); // object B is occluded frames 8-10

    let mut b_id_before: Option<u64> = None;
    let mut b_id_after: Option<u64> = None;

    for f in 0..=24 {
        let a_tlwh = [100.0 + 4.0 * f as f32, 100.0, 40.0, 80.0];
        let b_tlwh = [300.0 - 3.0 * f as f32, 120.0, 40.0, 80.0];

        let mut detections = vec![(a_tlwh, 0.9, 0)];
        let mut embeddings = vec![feat_a()];
        if !hidden(f) {
            detections.push((b_tlwh, 0.9, 1));
            embeddings.push(feat_b());
        }

        for t in tracker.update(detections.clone(), &embeddings) {
            let di = t.det_ind.expect("activated track must carry det_ind");
            assert!(di < detections.len());
            let recovered = &embeddings[di];
            if !is_a(recovered) {
                // This is object B's track.
                if f < 8 && b_id_before.is_none() {
                    b_id_before = Some(t.track_id);
                }
                if f > 12 && b_id_after.is_none() {
                    b_id_after = Some(t.track_id);
                }
            }
        }
    }

    // Object B must be tracked before and after the occlusion, and keep the same id
    // (re-acquired through Re-ID), with det_ind pointing at its detection again.
    let before = b_id_before.expect("B was never tracked before occlusion");
    let after = b_id_after.expect("B was not re-tracked after occlusion");
    assert_eq!(before, after, "B's track id changed across the occlusion");
}
