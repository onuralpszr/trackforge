use image::DynamicImage;
use std::error::Error;
use trackforge::trackers::deepsort::DeepSort;
use trackforge::traits::AppearanceExtractor;
use trackforge::types::BoundingBox;

struct MockExtractor;

impl AppearanceExtractor for MockExtractor {
    fn extract(
        &mut self,
        _image: &DynamicImage,
        bboxes: &[BoundingBox],
    ) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
        // For testing, return a mock embedding.
        // In real scenario, this would differ per object visual.
        // Here we just return a constant for simplicity, or random.
        // If we want separate tracks, we might want different embeddings.
        // Let's make it deterministic based on bbox center or something to match.

        let mut features = Vec::new();
        for bbox in bboxes {
            // Create a fake feature vector based on width/height to distinguish objects
            let v = vec![bbox.width / 100.0, bbox.height / 100.0];
            // Normalize
            let norm = (v[0].powi(2) + v[1].powi(2)).sqrt();
            let v = if norm > 0.0 {
                vec![v[0] / norm, v[1] / norm]
            } else {
                vec![0.0, 0.0]
            };
            features.push(v);
        }
        Ok(features)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let extractor = MockExtractor;
    let mut tracker = DeepSort::new_default(extractor);

    // Create a dummy image (never accessed by MockExtractor)
    let image = DynamicImage::new_rgb8(640, 480);

    // Simulated simple tracking scenario
    // Frame 1: Object A
    let detections_1 = vec![(BoundingBox::new(100.0, 100.0, 50.0, 100.0), 0.9, 0)];

    println!("Frame 1 processing...");
    let tracks_1 = tracker.update(&image, detections_1)?;
    println!("Frame 1 tracks: {:?}", tracks_1.len());
    for track in &tracks_1 {
        println!(" - Track ID: {}, State: {:?}", track.track_id, track.state);
    }

    // Frame 2: Object A moves slightly (should match)
    let detections_2 = vec![(BoundingBox::new(105.0, 100.0, 50.0, 100.0), 0.9, 0)];

    println!("Frame 2 processing...");
    let tracks_2 = tracker.update(&image, detections_2)?;
    println!("Frame 2 tracks: {:?}", tracks_2.len());
    for track in &tracks_2 {
        println!(" - Track ID: {}, State: {:?}", track.track_id, track.state);
    }

    // Frame 3: Object A again (confirms)
    let detections_3 = vec![(BoundingBox::new(110.0, 100.0, 50.0, 100.0), 0.9, 0)];
    println!("Frame 3 processing...");
    let tracks_3 = tracker.update(&image, detections_3)?;
    println!("Frame 3 tracks: {:?}", tracks_3.len());
    for track in &tracks_3 {
        println!(" - Track ID: {}, State: {:?}", track.track_id, track.state);
    }

    // Frame 4: Object B appears (new track)
    let detections_4 = vec![
        (BoundingBox::new(115.0, 100.0, 50.0, 100.0), 0.9, 0),
        (BoundingBox::new(300.0, 300.0, 60.0, 120.0), 0.8, 0),
    ];
    println!("Frame 4 processing...");
    let tracks_4 = tracker.update(&image, detections_4)?;
    println!("Frame 4 tracks: {:?}", tracks_4.len());
    for track in &tracks_4 {
        println!(" - Track ID: {}, State: {:?}", track.track_id, track.state);
    }

    Ok(())
}
