//! Usage example: mapping a track back to its source detection with `det_ind`.
//!
//! `det_ind` on every returned track tells you which detection in the current
//! frame's input list the track came from. Combined with a per-detection Re-ID
//! embedding, that lets you build a track-id -> embedding gallery for long-term
//! tracking: for each track, pull `detections[track.det_ind]` and reuse the
//! detection's appearance feature.
//!
//! Run with:  cargo run --example det_ind_demo

use std::collections::HashMap;

use trackforge::trackers::botsort::BotSort;

fn norm(v: &[f32]) -> Vec<f32> {
    let mag: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag > 1e-6 {
        v.iter().map(|x| x / mag).collect()
    } else {
        v.to_vec()
    }
}

fn main() {
    let mut tracker = BotSort::new(0.5, 30, 0.8, 0.6, 0.5, 0.25);

    // Two distinct identities (orthogonal Re-ID embeddings so they stay separate).
    let id_a = norm(&[1.0, 0.0, 0.0, 0.0]);
    let id_b = norm(&[0.0, 1.0, 0.0, 0.0]);

    // Per-track Re-ID gallery built *from the detections* via `det_ind`.
    let mut gallery: HashMap<u64, Vec<Vec<f32>>> = HashMap::new();

    for frame in 0..10 {
        // Current frame's detections: index = position in this vec.
        let detections = vec![
            ([100.0 + 5.0 * frame as f32, 100.0, 40.0, 80.0], 0.9, 0),
            ([300.0 - 3.0 * frame as f32, 120.0, 40.0, 80.0], 0.9, 1),
        ];
        let embeddings = vec![id_a.clone(), id_b.clone()];

        let tracks = tracker.update(detections.clone(), &embeddings);

        for track in &tracks {
            // `det_ind` maps this track back to its detection...
            let dt = track.det_ind.expect("track already has a source detection");
            let (src_box, _src_score, src_class) = &detections[dt];

            // ...so we can reuse the detection's Re-ID feature (e.g. append it to a
            // track-id gallery for long-term re-identification).
            let feature = &embeddings[dt];
            gallery
                .entry(track.track_id)
                .or_default()
                .push(feature.clone());

            println!(
                "frame {frame:2} | track {} from det {dt} (class {}) at {:?} | emb[..2] = {:?} | gallery len {}",
                track.track_id,
                src_class,
                src_box,
                &feature[..2],
                gallery[&track.track_id].len(),
            );
        }
    }

    // Each identity should have exactly one stable track id: the gallery lets you
    // find it again later purely by appearance.
    assert_eq!(gallery.len(), 2, "expected two distinct tracked identities");
    println!(
        "\nTwo stable track ids produced a per-id Re-ID gallery: {:?}",
        gallery.keys().collect::<Vec<_>>()
    );
}
