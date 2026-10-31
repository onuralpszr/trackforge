use trackforge::trackers::tracktrack::{TrackTrack, TrackTrackParams};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize TrackTrack with its defaults, then lower the confirmation length
    // so tracks are reported from the second frame.
    // det_thresh = 0.6: Score above which a detection is high confidence
    // match_thresh = 0.7: Association cost gate, lower is stricter
    // init_thresh = 0.7: Smallest score a leftover detection needs to start a track
    // tai_thresh = 0.55: Max IoU a new track may have with an existing one
    let mut params = TrackTrackParams::default();
    params.common.min_hits = 1;
    let mut tracker = TrackTrack::from_params(params);

    // Simulated detection input: [x, y, w, h], score, class_id.
    // The third box overlaps the first, so track-aware initialization skips it.
    let frames = vec![
        vec![
            ([100.0, 100.0, 50.0, 100.0], 0.9, 0),
            ([300.0, 200.0, 60.0, 120.0], 0.85, 0),
            ([105.0, 102.0, 50.0, 100.0], 0.75, 0),
        ],
        vec![
            ([104.0, 101.0, 50.0, 100.0], 0.92, 0), // Moved slightly
            ([303.0, 202.0, 60.0, 120.0], 0.88, 0),
        ],
        vec![
            ([108.0, 102.0, 50.0, 100.0], 0.91, 0),
            ([306.0, 204.0, 60.0, 120.0], 0.4, 0), // Low confidence, still matched
        ],
    ];

    for (i, detections) in frames.into_iter().enumerate() {
        println!("Processing Frame {}...", i + 1);
        // Pass embeddings for appearance-aware tracking, or an empty slice for motion only.
        for t in tracker.update(detections, &[]) {
            println!(
                "Track ID: {}, Box: {:?}, Score: {:.2}, det_ind: {:?}",
                t.track_id, t.tlwh, t.score, t.det_ind
            );
        }
        println!();
    }

    Ok(())
}
