//! Draw tracks on frames: a box and `#id class score` label per track, a short
//! motion trail, and a header line.

use std::collections::{HashMap, VecDeque};
use std::error::Error;

use ab_glyph::{FontArc, PxScale};
use image::{Rgb, RgbImage};
use imageproc::drawing::{
    draw_filled_rect_mut, draw_hollow_rect_mut, draw_line_segment_mut, draw_text_mut, text_size,
};
use imageproc::rect::Rect;

use crate::TrackOutput;

/// Ultralytics-style palette, picked by track id so an id keeps its color.
const PALETTE: [[u8; 3]; 10] = [
    [4, 42, 255],
    [11, 219, 235],
    [243, 243, 243],
    [0, 223, 183],
    [17, 31, 104],
    [255, 111, 221],
    [255, 68, 79],
    [204, 237, 0],
    [0, 243, 68],
    [189, 0, 255],
];

/// Number of past positions kept per track for its trail.
const TRAIL_LEN: usize = 40;

pub struct Annotator {
    font: FontArc,
    trails: HashMap<u64, (usize, VecDeque<(f32, f32)>)>,
    frame: usize,
}

impl Annotator {
    /// Load the Ultralytics font (downloaded once into the Ultralytics config dir).
    pub fn new() -> Result<Self, Box<dyn Error>> {
        let path = ultralytics_inference::annotate::check_font("Arial.ttf")
            .ok_or("could not load the Arial.ttf annotation font")?;
        let font = FontArc::try_from_vec(std::fs::read(path)?)?;
        Ok(Self {
            font,
            trails: HashMap::new(),
            frame: 0,
        })
    }

    /// Draw `tracks` and a `header` line onto `img`.
    pub fn draw(
        &mut self,
        img: &mut RgbImage,
        tracks: &[TrackOutput],
        label: impl Fn(i64) -> String,
        header: &str,
    ) {
        self.frame += 1;
        let (w, h) = img.dimensions();
        let thickness = (w.max(h) / 600).max(2) as i32;
        let scale = PxScale::from((h as f32 * 0.024).max(14.0));

        // Trails first so boxes and labels stay on top.
        for &(id, tlwh, _, _) in tracks {
            let (_, trail) = self
                .trails
                .entry(id)
                .or_insert_with(|| (self.frame, VecDeque::new()));
            trail.push_back((tlwh[0] + tlwh[2] / 2.0, tlwh[1] + tlwh[3]));
            if trail.len() > TRAIL_LEN {
                trail.pop_front();
            }
            self.trails.get_mut(&id).unwrap().0 = self.frame;
        }
        let frame = self.frame;
        self.trails
            .retain(|_, (seen, _)| frame - *seen <= TRAIL_LEN);
        for &(id, ..) in tracks {
            let color = color(id);
            let trail = &self.trails[&id].1;
            for (i, pair) in trail.iter().collect::<Vec<_>>().windows(2).enumerate() {
                // Older segments are thinner so the trail fades toward its tail.
                let t = 1 + (i as i32 * thickness) / TRAIL_LEN as i32;
                thick_line(img, *pair[0], *pair[1], t, color);
            }
        }

        for &(id, tlwh, score, class_id) in tracks {
            let color = color(id);
            let x = tlwh[0].max(0.0) as i32;
            let y = tlwh[1].max(0.0) as i32;
            let bw = (tlwh[2] as i32).clamp(1, w as i32 - x.min(w as i32 - 1));
            let bh = (tlwh[3] as i32).clamp(1, h as i32 - y.min(h as i32 - 1));
            for t in 0..thickness.min(bw / 2).min(bh / 2).max(1) {
                let r = Rect::at(x + t, y + t).of_size((bw - 2 * t) as u32, (bh - 2 * t) as u32);
                draw_hollow_rect_mut(img, r, color);
            }

            let text = format!("#{id} {} {score:.2}", label(class_id));
            let (tw, th) = text_size(scale, &self.font, &text);
            let pad = 3;
            let ly = (y - th as i32 - 2 * pad).max(0);
            let bg = Rect::at(x, ly).of_size(tw + 2 * pad as u32, th + 2 * pad as u32);
            draw_filled_rect_mut(img, bg, color);
            draw_text_mut(
                img,
                text_color(color),
                x + pad,
                ly + pad,
                scale,
                &self.font,
                &text,
            );
        }

        let (tw, th) = text_size(scale, &self.font, header);
        let pad = 8;
        let bar = Rect::at(0, 0).of_size(tw + 2 * pad as u32, th + 2 * pad as u32);
        draw_filled_rect_mut(img, bar, Rgb([20, 20, 28]));
        draw_text_mut(
            img,
            Rgb([255, 255, 255]),
            pad,
            pad,
            scale,
            &self.font,
            header,
        );
    }
}

fn color(id: u64) -> Rgb<u8> {
    Rgb(PALETTE[(id as usize) % PALETTE.len()])
}

/// Black text on light colors, white on dark ones.
fn text_color(bg: Rgb<u8>) -> Rgb<u8> {
    let [r, g, b] = bg.0;
    let luma = 0.299 * f32::from(r) + 0.587 * f32::from(g) + 0.114 * f32::from(b);
    if luma > 150.0 {
        Rgb([0, 0, 0])
    } else {
        Rgb([255, 255, 255])
    }
}

fn thick_line(img: &mut RgbImage, a: (f32, f32), b: (f32, f32), t: i32, color: Rgb<u8>) {
    for dx in -(t / 2)..=(t / 2) {
        for dy in -(t / 2)..=(t / 2) {
            let (ox, oy) = (dx as f32, dy as f32);
            draw_line_segment_mut(img, (a.0 + ox, a.1 + oy), (b.0 + ox, b.1 + oy), color);
        }
    }
}
