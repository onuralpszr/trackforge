//! Conversion helpers that turn detector output into tracker input.

/// Convert flat `[x1, y1, x2, y2, conf, cls]` detector output into tracker detections.
///
/// This is the layout most YOLO-style detectors produce after NMS, including the
/// `data` array of Ultralytics `Boxes`. Each group of six values becomes a
/// `(tlwh, score, class_id)` tuple, the input every tracker's `update` takes. Trailing
/// values that do not form a full group of six are ignored.
///
/// # Example
///
/// ```
/// use trackforge::utils::detections_from_xyxy;
///
/// let data = [10.0, 20.0, 50.0, 80.0, 0.9, 0.0];
/// let dets = detections_from_xyxy(&data);
/// assert_eq!(dets, vec![([10.0, 20.0, 40.0, 60.0], 0.9, 0)]);
/// ```
pub fn detections_from_xyxy(data: &[f32]) -> Vec<([f32; 4], f32, i64)> {
    let (boxes, _partial) = data.as_chunks::<6>();
    boxes
        .iter()
        .map(|b| ([b[0], b[1], b[2] - b[0], b[3] - b[1]], b[4], b[5] as i64))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detections_from_xyxy() {
        let data = [
            10.0, 20.0, 50.0, 80.0, 0.9, 0.0, //
            100.0, 100.0, 160.0, 220.0, 0.75, 2.0,
        ];
        let dets = detections_from_xyxy(&data);
        assert_eq!(
            dets,
            vec![
                ([10.0, 20.0, 40.0, 60.0], 0.9, 0),
                ([100.0, 100.0, 60.0, 120.0], 0.75, 2),
            ]
        );
    }

    #[test]
    fn test_detections_from_xyxy_empty_and_partial() {
        assert!(detections_from_xyxy(&[]).is_empty());
        assert!(detections_from_xyxy(&[1.0, 2.0, 3.0]).is_empty());
    }
}
