//! The whole loop in one screen: Ultralytics YOLO detects, trackforge tracks.
//!
//! ```bash
//! cargo run --release --features video --example minimal -- video.mp4
//! ```

use trackforge::trackers::byte_track::{ByteTrack, ByteTrackParams};
use trackforge::utils::detections_from_xyxy;
use ultralytics_inference::{Source, SourceIterator, YOLOModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let source = std::env::args().nth(1).ok_or("usage: minimal <video>")?;
    let mut model = YOLOModel::load("yolo26n.onnx")?;
    let mut tracker = ByteTrack::from_params(ByteTrackParams::default());

    for frame in SourceIterator::new(Source::from(source.as_str()))? {
        let (image, meta) = frame?;
        for result in model.predict_image(&image, meta.path)? {
            let Some(boxes) = &result.boxes else { continue };
            let data = boxes.data.as_standard_layout();
            let detections = detections_from_xyxy(data.as_slice().unwrap_or_default());
            for track in tracker.update(detections) {
                println!(
                    "frame {} id {} box {:?}",
                    meta.frame_idx, track.track_id, track.tlwh
                );
            }
        }
    }
    Ok(())
}
