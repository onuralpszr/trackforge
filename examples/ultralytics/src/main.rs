//! Track objects with Ultralytics YOLO inference in Rust and any trackforge tracker.
//!
//! ```bash
//! cd examples/ultralytics
//! # image folder or glob (frames in name order)
//! cargo run --release -- "frames/*.jpg"
//! # video file, webcam or stream (needs FFmpeg dev libraries)
//! cargo run --release --features video -- people.mp4 --tracker botsort --classes 0
//! ```
//!
//! Options: `--model <path>` (default `yolo26n.onnx`, downloaded on first use),
//! `--tracker <sort|bytetrack|ocsort|botsort|tracktrack>` (default `bytetrack`),
//! `--classes <id,id,...>` to keep only some classes.

use std::error::Error;

use trackforge::trackers::botsort::{BotSort, BotSortParams};
use trackforge::trackers::byte_track::{ByteTrack, ByteTrackParams};
use trackforge::trackers::ocsort::{OcSort, OcSortParams};
use trackforge::trackers::sort::{Sort, SortParams};
use trackforge::trackers::tracktrack::{TrackTrack, TrackTrackParams};
use trackforge::utils::detections_from_xyxy;
use ultralytics_inference::YOLOModel;
use ultralytics_inference::source::{Source, SourceIterator};

type Detections = Vec<([f32; 4], f32, i64)>;

/// One tracked object: `(track_id, tlwh, score, class_id)`.
type TrackOutput = (u64, [f32; 4], f32, i64);

/// The motion trackers, behind one `update` so the tracker can be picked at runtime.
enum Tracker {
    Sort(Sort),
    ByteTrack(ByteTrack),
    OcSort(OcSort),
    BotSort(BotSort),
    TrackTrack(TrackTrack),
}

impl Tracker {
    fn new(name: &str) -> Result<Self, String> {
        Ok(match name {
            "sort" => Self::Sort(Sort::from_params(SortParams::default())),
            "bytetrack" => Self::ByteTrack(ByteTrack::from_params(ByteTrackParams::default())),
            "ocsort" => Self::OcSort(OcSort::from_params(OcSortParams::default())),
            "botsort" => Self::BotSort(BotSort::from_params(BotSortParams::default())),
            "tracktrack" => Self::TrackTrack(TrackTrack::from_params(TrackTrackParams::default())),
            other => return Err(format!("unknown tracker: {other}")),
        })
    }

    fn update(&mut self, dets: Detections) -> Vec<TrackOutput> {
        macro_rules! collect_tracks {
            ($tracks:expr) => {
                $tracks
                    .into_iter()
                    .map(|t| (t.track_id, t.tlwh, t.score, t.class_id))
                    .collect()
            };
        }
        match self {
            Self::Sort(t) => collect_tracks!(t.update(dets)),
            Self::ByteTrack(t) => collect_tracks!(t.update(dets)),
            Self::OcSort(t) => collect_tracks!(t.update(dets)),
            // No Re-ID model here, so the appearance trackers run on motion alone.
            Self::BotSort(t) => collect_tracks!(t.update(dets, &[])),
            Self::TrackTrack(t) => collect_tracks!(t.update(dets, &[])),
        }
    }
}

struct Args {
    source: String,
    model: String,
    tracker: String,
    classes: Option<Vec<i64>>,
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let mut source = None;
    let mut model = "yolo26n.onnx".to_string();
    let mut tracker = "bytetrack".to_string();
    let mut classes = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model = args.next().ok_or("--model needs a value")?,
            "--tracker" => tracker = args.next().ok_or("--tracker needs a value")?,
            "--classes" => {
                let list = args.next().ok_or("--classes needs a value")?;
                classes = Some(
                    list.split(',')
                        .map(|c| c.trim().parse())
                        .collect::<Result<_, _>>()?,
                );
            }
            _ if source.is_none() => source = Some(arg),
            other => return Err(format!("unexpected argument: {other}").into()),
        }
    }
    let source = source.ok_or("usage: <source> [--model PATH] [--tracker NAME] [--classes 0,2]")?;
    Ok(Args {
        source,
        model,
        tracker,
        classes,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    // Silence the per-image log line so only the tracker output is printed.
    ultralytics_inference::logging::set_verbose(false);
    let mut model = YOLOModel::load(&args.model)?;
    let mut tracker = Tracker::new(&args.tracker)?;

    for frame in SourceIterator::new(Source::from(args.source.as_str()))? {
        let (image, meta) = frame?;
        let results = model.predict_image(&image, meta.path.clone())?;

        // Ultralytics `Boxes::data` holds one `[x1, y1, x2, y2, conf, cls]` entry per box.
        let mut dets = Vec::new();
        for result in &results {
            if let Some(boxes) = &result.boxes {
                let data = boxes.data.as_standard_layout();
                dets.extend(detections_from_xyxy(data.as_slice().unwrap_or_default()));
            }
        }
        if let Some(keep) = &args.classes {
            dets.retain(|d| keep.contains(&d.2));
        }

        let names = results.first().map(|r| r.names.clone());
        let tracks = tracker.update(dets);
        println!("frame {}: {} tracks", meta.frame_idx, tracks.len());
        for (id, tlwh, score, class_id) in tracks {
            let name = names
                .as_ref()
                .and_then(|n| n.get(&(class_id as usize)).cloned())
                .unwrap_or_else(|| class_id.to_string());
            println!(
                "  id {id:>3} {name:<10} {score:.2} [{:.0} {:.0} {:.0} {:.0}]",
                tlwh[0], tlwh[1], tlwh[2], tlwh[3]
            );
        }
    }
    Ok(())
}
