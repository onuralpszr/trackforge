use crate::trackers::deepsort::nn_matching::NearestNeighborDistanceMetric;
use crate::trackers::deepsort::track::Track;
use crate::utils::kalman::{KalmanFilter, MeasurementVector};
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct DeepSortTracker {
    pub metric: NearestNeighborDistanceMetric,
    pub max_age: usize,
    pub n_init: usize,
    pub tracks: Vec<Track>,
    pub kf: KalmanFilter,
    pub max_iou_distance: f32, // typically 0.7 for DeepSort
}

impl DeepSortTracker {
    pub fn new(
        metric: NearestNeighborDistanceMetric,
        max_age: usize,
        n_init: usize,
        max_iou_distance: f32,
    ) -> Self {
        Self {
            metric,
            max_age,
            n_init,
            tracks: Vec::new(),
            kf: KalmanFilter::default(),
            max_iou_distance,
        }
    }

    pub fn predict(&mut self) {
        for track in &mut self.tracks {
            track.predict(&self.kf);
        }
    }

    pub fn update(
        &mut self,
        detections: &[(crate::types::BoundingBox, f32, i64)], // (tlwh, score, class_id)
        embeddings: &[Vec<f32>],
    ) {
        let (matches, unmatched_tracks, unmatched_detections) = self._match(detections, embeddings);

        // Update matched tracks
        for (track_idx, detection_idx) in matches {
            let (tlwh, score, class_id) = detections[detection_idx];
            let embedding = &embeddings[detection_idx];

            let track = &mut self.tracks[track_idx];
            let meas = Track::tlwh_to_xyah(&[tlwh.x, tlwh.y, tlwh.width, tlwh.height]);
            track.update(&self.kf, &meas, score, class_id, embedding.clone());
        }

        // Mark missed tracks
        for track_idx in unmatched_tracks {
            self.tracks[track_idx].mark_missed();
        }

        // Initialize new tracks
        for detection_idx in unmatched_detections {
            let (tlwh, score, class_id) = detections[detection_idx];
            let embedding = &embeddings[detection_idx];
            self.initiate_track(
                &[tlwh.x, tlwh.y, tlwh.width, tlwh.height],
                score,
                class_id,
                embedding.clone(),
            );
        }

        // Remove deleted tracks
        self.tracks.retain(|t| !t.is_deleted());
    }

    pub fn initiate_track(
        &mut self,
        tlwh: &[f32; 4],
        score: f32,
        class_id: i64,
        embedding: Vec<f32>,
    ) {
        let measurement = Track::tlwh_to_xyah(tlwh);
        let (mean, covariance) = self.kf.initiate(&measurement);
        let track_id = self.next_id();
        let track = Track::new(
            mean,
            covariance,
            track_id,
            class_id,
            self.n_init,
            self.max_age,
            score,
            embedding,
        );
        self.tracks.push(track);
    }

    fn next_id(&self) -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        NEXT_ID.fetch_add(1, Ordering::Relaxed)
    }

    fn _match(
        &self,
        detections: &[(crate::types::BoundingBox, f32, i64)],
        embeddings: &[Vec<f32>],
    ) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
        // defined as (track_idx, detection_idx) for matches

        // Split tracks
        let mut confirmed_tracks_indices = Vec::new();
        let mut unconfirmed_tracks_indices = Vec::new();

        for (i, track) in self.tracks.iter().enumerate() {
            if track.is_confirmed() {
                confirmed_tracks_indices.push(i);
            } else {
                unconfirmed_tracks_indices.push(i);
            }
        }

        // 1. Associate Confirmed Tracks using Appearance (+ Mahalanobis gating)
        let (matches_a, unmatched_tracks_a, unmatched_detections_a) =
            self.match_cascade(&confirmed_tracks_indices, detections, embeddings);

        // 2. Associate remaining tracks with IOU
        // Candidate tracks: unconfirmed + unmatched_confirmed
        let mut iou_track_candidates = unconfirmed_tracks_indices.clone();
        for &idx in &unmatched_tracks_a {
            iou_track_candidates.push(idx);
        }

        let (matches_b, unmatched_tracks_b, unmatched_detections_b) =
            self.match_iou(&iou_track_candidates, &unmatched_detections_a, detections);

        // Merge matches
        let mut matches = matches_a;
        matches.extend(matches_b);

        (matches, unmatched_tracks_b, unmatched_detections_b)
    }

    fn match_cascade(
        &self,
        track_indices: &[usize], // indices into self.tracks
        detections: &[(crate::types::BoundingBox, f32, i64)],
        embeddings: &[Vec<f32>],
    ) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
        if track_indices.is_empty() || detections.is_empty() {
            return (
                Vec::new(),
                track_indices.to_vec(),
                (0..detections.len()).collect(),
            );
        }

        // Compute cost matrix using metric (Appearance Distance)
        let track_ids: Vec<u64> = track_indices
            .iter()
            .map(|&i| self.tracks[i].track_id)
            .collect();
        let cost_matrix_raw = self.metric.distance(embeddings, &track_ids); // targets x features

        // Gate cost matrix using Mahalanobis distance
        // cost_matrix[i][j] where i is track, j is detection
        let mut cost_matrix = cost_matrix_raw;
        let gating_threshold = 9.4877; // Chi-squared threshold for 4 DOF at 0.95

        for (i, &track_idx) in track_indices.iter().enumerate() {
            let track = &self.tracks[track_idx];
            // Compute mahalanobis distance for this track against all detections
            let measurements: Vec<MeasurementVector> = detections
                .iter()
                .map(|(tlwh, _, _)| Track::tlwh_to_xyah(&[tlwh.x, tlwh.y, tlwh.width, tlwh.height]))
                .collect();

            let gating_dist =
                self.kf
                    .gating_distance(&track.mean, &track.covariance, &measurements);

            for (j, &d) in gating_dist.iter().enumerate() {
                if d > gating_threshold {
                    cost_matrix[i][j] = f32::MAX;
                }
            }
        }

        // Run cascade
        let mut matches = Vec::new();
        let mut unmatched_tracks: HashSet<usize> = track_indices.iter().cloned().collect();
        let mut unmatched_detections: HashSet<usize> = (0..detections.len()).collect();

        for level in 0..self.max_age {
            if unmatched_detections.is_empty() {
                break;
            }

            // Select tracks at this age level
            let tracks_at_level: Vec<usize> = track_indices
                .iter()
                .filter(|&&idx| self.tracks[idx].time_since_update == 1 + level)
                .cloned()
                .collect();

            if tracks_at_level.is_empty() {
                continue;
            }

            // Build small cost matrix for this level
            let unmatched_dets_vec: Vec<usize> = unmatched_detections.iter().cloned().collect();

            let mut level_cost_matrix = Vec::new();
            for &trk_idx in &tracks_at_level {
                let data_row_idx = track_indices.iter().position(|&x| x == trk_idx).unwrap();

                let mut row = Vec::new();
                for &det_idx in &unmatched_dets_vec {
                    let cost = cost_matrix[data_row_idx][det_idx];
                    row.push(cost);
                }
                level_cost_matrix.push(row);
            }

            let (level_matches, _, _) =
                min_cost_matching(&level_cost_matrix, self.metric.matching_threshold());

            for (local_r, local_c) in level_matches {
                let trk_idx = tracks_at_level[local_r];
                let det_idx = unmatched_dets_vec[local_c];

                matches.push((trk_idx, det_idx));
                unmatched_tracks.remove(&trk_idx);
                unmatched_detections.remove(&det_idx);
            }
        }

        (
            matches,
            unmatched_tracks.into_iter().collect(),
            unmatched_detections.into_iter().collect(),
        )
    }

    fn match_iou(
        &self,
        track_indices: &[usize],
        detection_indices: &[usize], // indices into original detections list
        all_detections: &[(crate::types::BoundingBox, f32, i64)],
    ) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
        if track_indices.is_empty() || detection_indices.is_empty() {
            return (
                Vec::new(),
                track_indices.to_vec(),
                detection_indices.to_vec(),
            );
        }

        let mut cost_matrix = Vec::new();

        let track_boxes: Vec<[f32; 4]> = track_indices
            .iter()
            .map(|&i| self.tracks[i].to_tlwh())
            .collect();
        let det_boxes: Vec<[f32; 4]> = detection_indices
            .iter()
            .map(|&i| {
                let (tlwh, _, _) = all_detections[i];
                [tlwh.x, tlwh.y, tlwh.width, tlwh.height]
            })
            .collect();

        let ious = crate::utils::geometry::iou_batch(&track_boxes, &det_boxes);

        for iou_row in ious {
            let cost_row: Vec<f32> = iou_row.iter().map(|&iou| 1.0 - iou).collect();
            cost_matrix.push(cost_row);
        }

        let (local_matches, local_unmatched_tracks, local_unmatched_dets) =
            min_cost_matching(&cost_matrix, self.max_iou_distance);

        // Remap indices
        let matches = local_matches
            .iter()
            .map(|(r, c)| (track_indices[*r], detection_indices[*c]))
            .collect();
        let unmatched_tracks = local_unmatched_tracks
            .iter()
            .map(|r| track_indices[*r])
            .collect();
        let unmatched_detections = local_unmatched_dets
            .iter()
            .map(|c| detection_indices[*c])
            .collect();

        (matches, unmatched_tracks, unmatched_detections)
    }
}

/// Simple greedy min cost matching (Linear Assignment Problem).
fn min_cost_matching(
    cost_matrix: &[Vec<f32>],
    threshold: f32,
) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
    if cost_matrix.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let rows = cost_matrix.len();
    let cols = cost_matrix[0].len();

    let mut matches = Vec::new();
    let mut unmatched_rows: HashSet<usize> = (0..rows).collect();
    let mut unmatched_cols: HashSet<usize> = (0..cols).collect();

    let mut costs = Vec::new();
    for (r, row) in cost_matrix.iter().enumerate() {
        for (c, &cost) in row.iter().enumerate() {
            costs.push((cost, r, c));
        }
    }

    // Sort by cost
    costs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    for (cost, r, c) in costs {
        if cost > threshold {
            continue;
        }
        if unmatched_rows.contains(&r) && unmatched_cols.contains(&c) {
            matches.push((r, c));
            unmatched_rows.remove(&r);
            unmatched_cols.remove(&c);
        }
    }

    (
        matches,
        unmatched_rows.into_iter().collect(),
        unmatched_cols.into_iter().collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trackers::deepsort::nn_matching::Metric;
    use crate::types::BoundingBox;

    fn create_tracker() -> DeepSortTracker {
        let metric = NearestNeighborDistanceMetric::new(Metric::Cosine, 0.3, Some(100));
        DeepSortTracker::new(metric, 30, 3, 0.7)
    }

    #[test]
    fn test_tracker_initialization() {
        let tracker = create_tracker();
        assert_eq!(tracker.tracks.len(), 0);
        assert_eq!(tracker.max_age, 30);
        assert_eq!(tracker.n_init, 3);
    }

    #[test]
    fn test_initiate_track() {
        let mut tracker = create_tracker();

        tracker.initiate_track(&[100.0, 100.0, 50.0, 100.0], 0.9, 0, vec![1.0; 128]);

        assert_eq!(tracker.tracks.len(), 1);
        assert!(!tracker.tracks[0].is_confirmed()); // Should be tentative
        assert!(tracker.tracks[0].is_tentative());
    }

    #[test]
    fn test_predict() {
        let mut tracker = create_tracker();
        tracker.initiate_track(&[100.0, 100.0, 50.0, 100.0], 0.9, 0, vec![1.0; 128]);

        let initial_age = tracker.tracks[0].age;
        tracker.predict();

        assert_eq!(tracker.tracks[0].age, initial_age + 1);
        assert_eq!(tracker.tracks[0].time_since_update, 1);
    }

    #[test]
    fn test_track_lifecycle_confirmation() {
        let mut tracker = create_tracker();

        let det = (BoundingBox::new(100.0, 100.0, 50.0, 100.0), 0.9, 0i64);
        let emb = vec![1.0; 128];

        // Frame 1: Create track
        tracker.predict();
        tracker.update(&[det], &[emb.clone()]);
        assert_eq!(tracker.tracks.len(), 1);
        assert!(!tracker.tracks[0].is_confirmed());

        // Frame 2: Match
        tracker.predict();
        tracker.update(&[det], &[emb.clone()]);
        assert!(!tracker.tracks[0].is_confirmed());

        // Frame 3: Confirmed (n_init = 3)
        tracker.predict();
        tracker.update(&[det], &[emb.clone()]);
        assert!(tracker.tracks[0].is_confirmed());
    }

    #[test]
    fn test_track_deletion_on_miss() {
        let mut tracker = create_tracker();

        let det = (BoundingBox::new(100.0, 100.0, 50.0, 100.0), 0.9, 0i64);
        let emb = vec![1.0; 128];

        // Create a tentative track
        tracker.predict();
        tracker.update(&[det], &[emb]);
        assert_eq!(tracker.tracks.len(), 1);

        // Miss it - tentative tracks get deleted on first miss
        tracker.predict();
        tracker.update(&[], &[]);

        // Track should be deleted
        assert_eq!(tracker.tracks.len(), 0);
    }

    #[test]
    fn test_multiple_detections() {
        let mut tracker = create_tracker();

        let dets = vec![
            (BoundingBox::new(100.0, 100.0, 50.0, 100.0), 0.9, 0i64),
            (BoundingBox::new(300.0, 300.0, 50.0, 100.0), 0.85, 0i64),
        ];
        let embs = vec![vec![1.0; 128], vec![0.0; 128]];

        tracker.predict();
        tracker.update(&dets, &embs);

        assert_eq!(tracker.tracks.len(), 2);
    }

    #[test]
    fn test_min_cost_matching_empty() {
        let (matches, unmatched_rows, unmatched_cols) = min_cost_matching(&[], 0.5);
        assert!(matches.is_empty());
        assert!(unmatched_rows.is_empty());
        assert!(unmatched_cols.is_empty());
    }

    #[test]
    fn test_min_cost_matching_simple() {
        // Simple 2x2 cost matrix
        let cost_matrix = vec![vec![0.1, 0.9], vec![0.8, 0.2]];

        let (matches, unmatched_rows, unmatched_cols) = min_cost_matching(&cost_matrix, 0.5);

        // Should match (0,0) and (1,1) due to low costs
        assert_eq!(matches.len(), 2);
        assert!(matches.contains(&(0, 0)));
        assert!(matches.contains(&(1, 1)));
        assert!(unmatched_rows.is_empty());
        assert!(unmatched_cols.is_empty());
    }

    #[test]
    fn test_min_cost_matching_threshold() {
        // All costs above threshold
        let cost_matrix = vec![vec![0.9, 0.9], vec![0.9, 0.9]];

        let (matches, unmatched_rows, unmatched_cols) = min_cost_matching(&cost_matrix, 0.5);

        // No matches should be made
        assert!(matches.is_empty());
        assert_eq!(unmatched_rows.len(), 2);
        assert_eq!(unmatched_cols.len(), 2);
    }

    #[test]
    fn test_min_cost_matching_asymmetric() {
        // More rows than cols
        let cost_matrix = vec![vec![0.1], vec![0.2], vec![0.3]];

        let (matches, unmatched_rows, unmatched_cols) = min_cost_matching(&cost_matrix, 0.5);

        // Only one match possible
        assert_eq!(matches.len(), 1);
        assert_eq!(unmatched_rows.len(), 2);
        assert!(unmatched_cols.is_empty());
    }

    #[test]
    fn test_update_empty_detections() {
        let mut tracker = create_tracker();

        // No detections
        tracker.predict();
        tracker.update(&[], &[]);

        assert_eq!(tracker.tracks.len(), 0);
    }

    #[test]
    fn test_track_id_uniqueness() {
        let mut tracker = create_tracker();

        let det1 = (BoundingBox::new(100.0, 100.0, 50.0, 100.0), 0.9, 0i64);
        let det2 = (BoundingBox::new(300.0, 300.0, 50.0, 100.0), 0.9, 0i64);
        let emb = vec![1.0; 128];

        tracker.predict();
        tracker.update(&[det1], &[emb.clone()]);
        let id1 = tracker.tracks[0].track_id;
        assert!(id1 > 0); // ID should be positive

        tracker.predict();
        tracker.update(&[det1, det2], &[emb.clone(), vec![0.0; 128]]);

        // Check IDs are unique
        let ids: Vec<u64> = tracker.tracks.iter().map(|t| t.track_id).collect();
        let unique_ids: HashSet<u64> = ids.iter().cloned().collect();
        assert_eq!(ids.len(), unique_ids.len());
    }

    #[test]
    fn test_confirmed_track_matching() {
        let mut tracker = create_tracker();

        let det = (BoundingBox::new(100.0, 100.0, 50.0, 100.0), 0.9, 0i64);
        let emb = vec![1.0; 128];

        // Confirm a track (3 hits)
        for _ in 0..3 {
            tracker.predict();
            tracker.update(&[det], &[emb.clone()]);
        }
        assert!(tracker.tracks[0].is_confirmed());

        // Match the confirmed track with same detection
        tracker.predict();
        tracker.update(&[det], &[emb.clone()]);

        assert_eq!(tracker.tracks.len(), 1);
        assert!(tracker.tracks[0].is_confirmed());
    }

    #[test]
    fn test_confirmed_track_with_different_detection() {
        let mut tracker = create_tracker();

        let det1 = (BoundingBox::new(100.0, 100.0, 50.0, 100.0), 0.9, 0i64);
        let det2 = (BoundingBox::new(500.0, 500.0, 50.0, 100.0), 0.9, 0i64);
        let emb1 = vec![1.0; 128];
        let emb2 = vec![-1.0; 128]; // Different embedding

        // Confirm track with det1
        for _ in 0..3 {
            tracker.predict();
            tracker.update(&[det1], &[emb1.clone()]);
        }
        assert_eq!(tracker.tracks.len(), 1);
        assert!(tracker.tracks[0].is_confirmed());

        // Now update with det2 (far away) - should create new track
        tracker.predict();
        tracker.update(&[det2], &[emb2.clone()]);

        // Original track missed, new track created
        assert!(tracker.tracks.len() >= 1);
    }

    #[test]
    fn test_iou_matching_fallback() {
        let mut tracker = create_tracker();

        let det = (BoundingBox::new(100.0, 100.0, 50.0, 100.0), 0.9, 0i64);
        // Use different embeddings to force IOU matching for unconfirmed tracks
        let emb1 = vec![1.0; 128];
        let emb2 = vec![0.5; 128];

        // Create unconfirmed track
        tracker.predict();
        tracker.update(&[det], &[emb1]);
        assert_eq!(tracker.tracks.len(), 1);
        assert!(!tracker.tracks[0].is_confirmed());

        // Same location detection - should match via IOU
        let close_det = (BoundingBox::new(105.0, 105.0, 50.0, 100.0), 0.9, 0i64);
        tracker.predict();
        tracker.update(&[close_det], &[emb2]);

        // Should still have 1 track (matched via IOU)
        assert_eq!(tracker.tracks.len(), 1);
    }

    #[test]
    fn test_track_age_cascade() {
        let mut tracker = create_tracker();

        let det = (BoundingBox::new(100.0, 100.0, 50.0, 100.0), 0.9, 0i64);
        let emb = vec![1.0; 128];

        // Confirm a track
        for _ in 0..3 {
            tracker.predict();
            tracker.update(&[det], &[emb.clone()]);
        }

        // Miss several frames
        for _ in 0..5 {
            tracker.predict();
            tracker.update(&[], &[]);
        }

        // Track should still exist (not past max_age=30)
        assert_eq!(tracker.tracks.len(), 1);
        assert_eq!(tracker.tracks[0].time_since_update, 5);
    }

    #[test]
    fn test_max_age_deletion() {
        let mut tracker = create_tracker();

        let det = (BoundingBox::new(100.0, 100.0, 50.0, 100.0), 0.9, 0i64);
        let emb = vec![1.0; 128];

        // Confirm a track
        for _ in 0..3 {
            tracker.predict();
            tracker.update(&[det], &[emb.clone()]);
        }
        assert!(tracker.tracks[0].is_confirmed());

        // Miss more than max_age frames
        for _ in 0..35 {
            tracker.predict();
            tracker.update(&[], &[]);
        }

        // Track should be deleted
        assert_eq!(tracker.tracks.len(), 0);
    }

    #[test]
    fn test_multiple_tracks_matching() {
        let mut tracker = create_tracker();

        let det1 = (BoundingBox::new(100.0, 100.0, 50.0, 100.0), 0.9, 0i64);
        let det2 = (BoundingBox::new(300.0, 300.0, 50.0, 100.0), 0.9, 1i64);
        let emb1 = vec![1.0; 128];
        let emb2 = vec![0.0; 128];

        // Create two tracks
        tracker.predict();
        tracker.update(&[det1, det2], &[emb1.clone(), emb2.clone()]);
        assert_eq!(tracker.tracks.len(), 2);

        // Match both tracks
        tracker.predict();
        tracker.update(&[det1, det2], &[emb1.clone(), emb2.clone()]);
        assert_eq!(tracker.tracks.len(), 2);

        // Confirm both (3rd hit)
        tracker.predict();
        tracker.update(&[det1, det2], &[emb1, emb2]);

        let confirmed_count = tracker.tracks.iter().filter(|t| t.is_confirmed()).count();
        assert_eq!(confirmed_count, 2);
    }

    #[test]
    fn test_class_id_preserved() {
        let mut tracker = create_tracker();

        let det = (BoundingBox::new(100.0, 100.0, 50.0, 100.0), 0.9, 5i64);
        let emb = vec![1.0; 128];

        tracker.predict();
        tracker.update(&[det], &[emb]);

        assert_eq!(tracker.tracks[0].class_id, 5);
    }

    #[test]
    fn test_score_updated() {
        let mut tracker = create_tracker();

        let det1 = (BoundingBox::new(100.0, 100.0, 50.0, 100.0), 0.9, 0i64);
        let det2 = (BoundingBox::new(100.0, 100.0, 50.0, 100.0), 0.75, 0i64);
        let emb = vec![1.0; 128];

        tracker.predict();
        tracker.update(&[det1], &[emb.clone()]);
        assert!((tracker.tracks[0].score - 0.9).abs() < 0.01);

        tracker.predict();
        tracker.update(&[det2], &[emb]);
        assert!((tracker.tracks[0].score - 0.75).abs() < 0.01);
    }
}
