//! Track objects with Ultralytics YOLO inference in Rust and any trackforge tracker.
//!
//! ```bash
//! cd examples/ultralytics
//! # image folder or glob (frames in name order)
//! cargo run --release -- "frames/*.jpg"
//! # video file, webcam or stream (needs FFmpeg dev libraries)
//! cargo run --release --features video -- people.mp4 --tracker botsort --classes 0
//! # write an annotated video
//! cargo run --release --features video -- people.mp4 --classes 0 --save tracked.mp4
//! ```
//!
//! Options: `--model <path>` (default `yolo26n.onnx`, downloaded on first use),
//! `--tracker <sort|bytetrack|ocsort|botsort|tracktrack>` (default `bytetrack`),
//! `--classes <id,id,...>` to keep only some classes, `--conf <0..1>` and `--imgsz <px>`
//! for the detector, and `--save <out.mp4>` to write an annotated video with track ids
//! and motion trails (needs `--features video`).

mod annotate;

use std::error::Error;

use trackforge::trackers::botsort::{BotSort, BotSortParams};
use trackforge::trackers::byte_track::{ByteTrack, ByteTrackParams};
use trackforge::trackers::ocsort::{OcSort, OcSortParams};
use trackforge::trackers::sort::{Sort, SortParams};
use trackforge::trackers::tracktrack::{TrackTrack, TrackTrackParams};
use trackforge::utils::detections_from_xyxy;

use annotate::Annotator;
use ultralytics_inference::source::{Source, SourceIterator};
use ultralytics_inference::{InferenceConfig, YOLOModel};

type Detections = Vec<([f32; 4], f32, i64)>;

/// One tracked object: `(track_id, tlwh, score, class_id)`.
pub(crate) type TrackOutput = (u64, [f32; 4], f32, i64);

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
    conf: Option<f32>,
    imgsz: Option<usize>,
    save: Option<String>,
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let mut parsed = Args {
        source: String::new(),
        model: "yolo26n.onnx".to_string(),
        tracker: "bytetrack".to_string(),
        classes: None,
        conf: None,
        imgsz: None,
        save: None,
    };
    let mut source = None;
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or(format!("{arg} needs a value"));
        match arg.as_str() {
            "--model" => parsed.model = value()?,
            "--tracker" => parsed.tracker = value()?,
            "--classes" => {
                parsed.classes = Some(
                    value()?
                        .split(',')
                        .map(|c| c.trim().parse())
                        .collect::<Result<_, _>>()?,
                );
            }
            "--conf" => parsed.conf = Some(value()?.parse()?),
            "--imgsz" => parsed.imgsz = Some(value()?.parse()?),
            "--save" => parsed.save = Some(value()?),
            _ if source.is_none() => source = Some(arg),
            other => return Err(format!("unexpected argument: {other}").into()),
        }
    }
    parsed.source = source.ok_or(
        "usage: <source> [--model PATH] [--tracker NAME] [--classes 0,2] [--conf 0.25] [--imgsz 640] [--save out.mp4]",
    )?;
    Ok(parsed)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    if args.save.is_some() && !cfg!(feature = "video") {
        return Err("--save writes a video, so build with --features video".into());
    }
    // Silence the per-image log line so only the tracker output is printed.
    ultralytics_inference::logging::set_verbose(false);

    let mut config = InferenceConfig::new();
    if let Some(conf) = args.conf {
        config = config.with_confidence(conf);
    }
    if let Some(imgsz) = args.imgsz {
        config = config.with_imgsz(imgsz, imgsz);
    }
    let mut model = YOLOModel::load_with_config(&args.model, config)?;
    let mut tracker = Tracker::new(&args.tracker)?;
    let mut annotator = args.save.as_ref().map(|_| Annotator::new()).transpose()?;
    #[cfg(feature = "video")]
    let mut writer: Option<ultralytics_inference::io::VideoWriter> = None;

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
        let name_of = |class_id: i64| {
            names
                .as_ref()
                .and_then(|n| n.get(&(class_id as usize)).cloned())
                .unwrap_or_else(|| class_id.to_string())
        };
        let tracks = tracker.update(dets);
        println!("frame {}: {} tracks", meta.frame_idx, tracks.len());

        let Some(annotator) = annotator.as_mut() else {
            for &(id, tlwh, score, class_id) in &tracks {
                println!(
                    "  id {id:>3} {:<10} {score:.2} [{:.0} {:.0} {:.0} {:.0}]",
                    name_of(class_id),
                    tlwh[0],
                    tlwh[1],
                    tlwh[2],
                    tlwh[3]
                );
            }
            continue;
        };

        let mut canvas = image.to_rgb8();
        let header = format!(
            "trackforge {} + ultralytics-inference | frame {} | {} tracks",
            args.tracker,
            meta.frame_idx + 1,
            tracks.len()
        );
        annotator.draw(&mut canvas, &tracks, name_of, &header);

        #[cfg(feature = "video")]
        {
            let writer = match writer.as_mut() {
                Some(writer) => writer,
                None => writer.insert(ultralytics_inference::io::VideoWriter::new(
                    args.save.as_deref().unwrap_or("tracked.mp4"),
                    canvas.width() as usize,
                    canvas.height() as usize,
                    meta.fps.unwrap_or(30.0),
                )?),
            };
            writer.write_frame(&image::DynamicImage::ImageRgb8(canvas))?;
        }
    }

    #[cfg(feature = "video")]
    if let Some(writer) = writer {
        writer.finish()?;
        println!("saved {}", args.save.as_deref().unwrap_or_default());
    }
    Ok(())
}
