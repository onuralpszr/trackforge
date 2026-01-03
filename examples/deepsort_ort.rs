use image::DynamicImage;
use opencv::{
    core::{Mat, Point, Rect, Scalar, Size},
    imgproc,
    prelude::*,
    videoio::{self, VideoCapture, VideoWriter},
};
use ort::session::{Session, builder::GraphOptimizationLevel};
use std::error::Error;
use std::time::Instant;
use trackforge::trackers::deepsort::DeepSort;
use trackforge::traits::AppearanceExtractor;
use trackforge::types::BoundingBox;
use usls::{Config, DType, models::RTDETR};

// Simple ReID Extractor using ORT
struct ReIDExtractor {
    session: Session,
}

impl ReIDExtractor {
    fn new(model_path: &str) -> Result<Self, Box<dyn Error>> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;
        Ok(Self { session })
    }

    fn preprocess(&self, image: &DynamicImage) -> ndarray::Array4<f32> {
        // Resize to model input 224x224
        let resized = image.resize_exact(224, 224, image::imageops::FilterType::Triangle);
        let rgb = resized.to_rgb8();

        // Normalize (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        let mean = [0.485, 0.456, 0.406];
        let std = [0.229, 0.224, 0.225];

        let mut array = ndarray::Array4::<f32>::zeros((1, 3, 224, 224));
        for (x, y, pixel) in rgb.enumerate_pixels() {
            let (x, y) = (x as usize, y as usize);
            let [r, g, b] = pixel.0;
            array[[0, 0, y, x]] = (r as f32 / 255.0 - mean[0]) / std[0];
            array[[0, 1, y, x]] = (g as f32 / 255.0 - mean[1]) / std[1];
            array[[0, 2, y, x]] = (b as f32 / 255.0 - mean[2]) / std[2];
        }
        array
    }
}

impl AppearanceExtractor for ReIDExtractor {
    fn extract(
        &mut self,
        image: &DynamicImage,
        bboxes: &[BoundingBox],
    ) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
        let mut embeddings = Vec::new();

        for bbox in bboxes {
            // Crop
            let crop = image.crop_imm(
                bbox.x as u32,
                bbox.y as u32,
                bbox.width as u32,
                bbox.height as u32,
            );

            // Preprocess
            let input = self.preprocess(&crop);

            // Basic inference
            // Basic inference
            let shape = input.shape().to_vec();
            let data = input.into_raw_vec_and_offset().0;
            let input_tensor = ort::value::Tensor::from_array((shape, data))?;

            // Get input name dynamically if possible, or assume input[0]
            let input_name = self.session.inputs[0].name.clone();

            let outputs = self
                .session
                .run(ort::inputs![input_name.as_str() => input_tensor])?;
            let output_tuple = outputs[0].try_extract_tensor::<f32>()?;
            let output_slice = output_tuple.1;

            // Convert to Vec<f32>
            let embedding: Vec<f32> = output_slice.iter().cloned().collect();

            // Normalize (L2) - optional but critical for DeepSort cosine distance
            let norm: f32 = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
            let normalized = if norm > 1e-6 {
                embedding.iter().map(|v| v / norm).collect()
            } else {
                embedding
            };

            embeddings.push(normalized);
        }
        Ok(embeddings)
    }
}

// enum Detector removed

// impl Detector {
//     fn run(&mut self, img: &DynamicImage) -> Result<Vec<(BoundingBox, f32, i64)>, Box<dyn Error>> {
//         // Placeholder for USLS integration
//         // let images = vec![img.clone()];
//         // let detections = self.yolo.run(&images)?;
//         Ok(vec![])
//     }
// }

// Mock Detector for Example compilation
struct Detector {
    model: RTDETR,
}

impl Detector {
    fn new(model_path: &str) -> Result<Self, Box<dyn Error>> {
        let config = Config::rtdetr()
            .with_model_file(model_path)
            .with_dtype_all("f32".parse::<DType>()?)
            .commit()?;
        let model = RTDETR::new(config)?;
        Ok(Self { model })
    }

    fn run(&mut self, img: &DynamicImage) -> Result<Vec<(BoundingBox, f32, i64)>, Box<dyn Error>> {
        let image: usls::Image = img.clone().into();
        let xs = vec![image];
        let ys = self.model.forward(&xs)?;

        let mut result = Vec::new();
        if let Some(y) = ys.first() {
            for obj in &y.hbbs {
                let x = obj.xmin();
                let y = obj.ymin();
                let w = obj.xmax() - obj.xmin();
                let h = obj.ymax() - obj.ymin();
                let conf = obj.confidence().unwrap_or(0.0);
                let cls = obj.id().unwrap_or(0);

                let bbox = BoundingBox::new(x, y, w, h);
                result.push((bbox, conf, cls as i64));
            }
        }
        Ok(result)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        println!("Usage: deepsort_ort <video_path> <model_path> <reid_path> <output_path>");
        return Ok(());
    }
    let video_path = &args[1];
    let model_path = &args[2];
    let reid_path = &args[3];
    let output_path = &args[4];

    // Load models
    println!("Loading models...");

    // Use Real Detector
    let mut detector = Detector::new(model_path)?;
    let extractor = ReIDExtractor::new(reid_path)?;

    // Init tracker
    let mut tracker = DeepSort::new(extractor, 60, 3, 0.7, 0.2, 100);

    // Open Video
    let mut cam = VideoCapture::from_file(video_path, videoio::CAP_ANY)?;
    if !cam.is_opened()? {
        return Err("Unable to open video file".into());
    }

    let width = cam.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
    let height = cam.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;
    let fps = cam.get(videoio::CAP_PROP_FPS)?;
    let frame_count_total = cam.get(videoio::CAP_PROP_FRAME_COUNT)? as i32;
    println!(
        "Video info: {}x{} @ {}fps, {} frames",
        width, height, fps, frame_count_total
    );

    let mut writer = VideoWriter::new(
        output_path,
        VideoWriter::fourcc('a', 'v', 'c', '1')?,
        fps,
        Size::new(width, height),
        true,
    )?;

    println!("Processing video...");
    let mut frame = Mat::default();
    let mut frame_count = 0;

    loop {
        if frame_count > 300 {
            println!("Reached frame limit (300), stopping.");
            break;
        }
        if !cam.read(&mut frame)? {
            break;
        }
        if frame.empty() {
            continue;
        }

        // Convert Mat to DynamicImage
        // OpenCV is BGR, Image is RGB usually
        let mut rgb = Mat::default();
        imgproc::cvt_color(
            &frame,
            &mut rgb,
            imgproc::COLOR_BGR2RGB,
            0,
            opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
        let data = rgb.data_bytes()?;
        let img = image::RgbImage::from_raw(width as u32, height as u32, data.to_vec())
            .ok_or("Failed to create image")?;
        let dyn_img = DynamicImage::ImageRgb8(img);

        let start = Instant::now();

        // 1. Detect
        let detections = detector.run(&dyn_img)?;

        // 2. Track
        let tracks = tracker.update(&dyn_img, detections)?;

        let duration = start.elapsed();

        // 3. Visualize
        for track in &tracks {
            let tlwh = track.to_tlwh();
            let rect = Rect::new(
                tlwh[0] as i32,
                tlwh[1] as i32,
                tlwh[2] as i32,
                tlwh[3] as i32,
            );
            let color = Scalar::new(0.0, 255.0, 0.0, 0.0);

            imgproc::rectangle(&mut frame, rect, color, 2, imgproc::LINE_8, 0)?;

            let label = format!("ID: {}", track.track_id);
            imgproc::put_text(
                &mut frame,
                &label,
                Point::new(rect.x, rect.y - 5),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                imgproc::LINE_8,
                false,
            )?;
        }

        // FPS
        imgproc::put_text(
            &mut frame,
            &format!("FPS: {:.1}", 1.0 / duration.as_secs_f32()),
            Point::new(10, 30),
            imgproc::FONT_HERSHEY_SIMPLEX,
            1.0,
            Scalar::new(0.0, 0.0, 255.0, 0.0),
            2,
            imgproc::LINE_8,
            false,
        )?;

        println!("Frame {}: {} tracks", frame_count, tracks.len());
        writer.write(&frame)?;
        frame_count += 1;
    }

    println!("Done! Output saved to {}", output_path);
    Ok(())
}
