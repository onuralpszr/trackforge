//! Conversion helpers that turn detector output into tracker input.

/// Convert flat `[x1, y1, x2, y2, conf, cls]` detection rows into tracker detections.
///
/// This is the row layout most YOLO-style detectors produce after NMS, including
/// the `data` array of Ultralytics `Boxes`. Each row becomes a `(tlwh, score, class_id)`
/// tuple, the input every tracker's `update` takes. A trailing partial row is ignored.
///
/// # Example
///
/// ```
/// use trackforge::utils::detections_from_xyxy_rows;
///
/// let rows = [10.0, 20.0, 50.0, 80.0, 0.9, 0.0];
/// let dets = detections_from_xyxy_rows(&rows);
/// assert_eq!(dets, vec![([10.0, 20.0, 40.0, 60.0], 0.9, 0)]);
/// ```
pub fn detections_from_xyxy_rows(rows: &[f32]) -> Vec<([f32; 4], f32, i64)> {
    let (rows, _partial) = rows.as_chunks::<6>();
    rows.iter()
        .map(|r| ([r[0], r[1], r[2] - r[0], r[3] - r[1]], r[4], r[5] as i64))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detections_from_xyxy_rows() {
        let rows = [
            10.0, 20.0, 50.0, 80.0, 0.9, 0.0, //
            100.0, 100.0, 160.0, 220.0, 0.75, 2.0,
        ];
        let dets = detections_from_xyxy_rows(&rows);
        assert_eq!(
            dets,
            vec![
                ([10.0, 20.0, 40.0, 60.0], 0.9, 0),
                ([100.0, 100.0, 60.0, 120.0], 0.75, 2),
            ]
        );
    }

    #[test]
    fn test_detections_from_xyxy_rows_empty_and_partial() {
        assert!(detections_from_xyxy_rows(&[]).is_empty());
        assert!(detections_from_xyxy_rows(&[1.0, 2.0, 3.0]).is_empty());
    }
}
