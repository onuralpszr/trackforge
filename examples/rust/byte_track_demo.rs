use anyhow::Result;
use trackforge::trackers::byte_track::ByteTrack;

fn main() -> Result<()> {
    // Initialize ByteTrack
    // track_thresh = 0.5: Threshold for high confidence detections
    // track_buffer = 30: Frames to keep lost tracks alive
    // match_thresh = 0.8: IoU threshold for matching
    // det_thresh = 0.6: Threshold for detection initialization
    let mut tracker = ByteTrack::new(0.5, 30, 0.8, 0.6);

    // Simulated detection input: [x, y, w, h], score, class_id
    let frame_1_detections = vec![
        ([100.0, 100.0, 50.0, 100.0], 0.9, 0),
        ([200.0, 200.0, 60.0, 120.0], 0.85, 0),
    ];

    println!("Processing Frame 1...");
    let tracks_1 = tracker.update(frame_1_detections)?;

    for t in tracks_1 {
        println!(
            "Track ID: {}, Box: {:?}, Score: {:.2}",
            t.track_id, t.tlwh, t.score
        );
    }

    // Simulated movement for Frame 2
    let frame_2_detections = vec![
        ([105.0, 102.0, 50.0, 100.0], 0.92, 0), // Moved slightly
        ([202.0, 201.0, 60.0, 120.0], 0.88, 0),
    ];

    println!("\nProcessing Frame 2...");
    let tracks_2 = tracker.update(frame_2_detections)?;

    for t in tracks_2 {
        println!(
            "Track ID: {}, Box: {:?}, Score: {:.2}",
            t.track_id, t.tlwh, t.score
        );
    }

    Ok(())
}
