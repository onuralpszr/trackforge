//! Deep OC-SORT association: OC-SORT motion plus an appearance affinity term.

use crate::trackers::common::association::{last_observation_rematch, ocm_angle_bonus};
use crate::trackers::common::{CameraMotion, ObsTrack as DeepOcSortTrack, TrackState};
use crate::trackers::deepsort::NearestNeighborDistanceMetric;
use crate::utils::assignment::greedy_match;
use crate::utils::geometry::{iou_batch, tlwh_to_xyah};
use crate::utils::kalman::KalmanFilter;
use std::collections::HashSet;

/// A detection paired with its optional appearance embedding.
struct Detection {
    tlwh: [f32; 4],
    score: f32,
    class_id: i64,
}

/// Deep OC-SORT tracker core.
///
/// Runs the OC-SORT motion association (IoU with an OCM direction bonus, plus an
/// ORU re-update on re-association) and blends in an appearance cost from a cosine
/// feature gallery. The appearance weight scales with detector confidence (dynamic
/// appearance) and is gated by `max_cosine_distance`. With `appearance_weight = 0`
/// the association reduces to plain OC-SORT.
pub struct DeepOcSortTracker {
    pub tracks: Vec<DeepOcSortTrack>,
    max_age: usize,
    min_hits: usize,
    iou_threshold: f32,
    delta_t: usize,
    inertia: f32,
    appearance_weight: f32,
    max_cosine_distance: f32,
    metric: NearestNeighborDistanceMetric,
    kf: KalmanFilter,
    next_id: u64,
    frame_count: usize,
}

impl DeepOcSortTracker {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        max_age: usize,
        min_hits: usize,
        iou_threshold: f32,
        delta_t: usize,
        inertia: f32,
        appearance_weight: f32,
        max_cosine_distance: f32,
        metric: NearestNeighborDistanceMetric,
    ) -> Self {
        Self {
            tracks: Vec::new(),
            max_age,
            min_hits,
            iou_threshold,
            delta_t,
            inertia: inertia.clamp(0.0, 1.0),
            appearance_weight: appearance_weight.clamp(0.0, 1.0),
            max_cosine_distance,
            metric,
            kf: KalmanFilter::default(),
            next_id: 1,
            frame_count: 0,
        }
    }

    /// Update the tracker with the current frame's detections and embeddings.
    ///
    /// `embeddings` is parallel to `detections`. Pass an empty slice to run without
    /// appearance (pure OC-SORT). Returns confirmed tracks matched this frame.
    pub fn update(
        &mut self,
        detections: &[([f32; 4], f32, i64)],
        embeddings: &[Vec<f32>],
    ) -> Vec<DeepOcSortTrack> {
        self.update_with_camera_motion(detections, embeddings, &CameraMotion::identity())
    }

    /// Update the tracker, first warping track predictions by `camera_motion`.
    ///
    /// `camera_motion` maps the previous frame's coordinates into the current frame
    /// (see [`CameraMotion`]); pass [`CameraMotion::identity`] for a static camera.
    pub fn update_with_camera_motion(
        &mut self,
        detections: &[([f32; 4], f32, i64)],
        embeddings: &[Vec<f32>],
        camera_motion: &CameraMotion,
    ) -> Vec<DeepOcSortTrack> {
        self.frame_count += 1;

        let use_appearance = !embeddings.is_empty() && embeddings.len() == detections.len();
        let dets: Vec<Detection> = detections
            .iter()
            .map(|(tlwh, score, class_id)| Detection {
                tlwh: *tlwh,
                score: *score,
                class_id: *class_id,
            })
            .collect();

        let warp = !camera_motion.is_identity();
        for track in &mut self.tracks {
            track.predict(&self.kf);
            if warp {
                track.apply_camera_motion(camera_motion);
            }
        }

        let (matches, unmatched_dets, unmatched_trks) =
            self.associate(&dets, embeddings, use_appearance);

        for (det_idx, trk_idx) in &matches {
            let det = &dets[*det_idx];
            let xyah = tlwh_to_xyah(&det.tlwh);
            let track = &mut self.tracks[*trk_idx];

            if track.time_since_update > 0 {
                track.our_re_update(&xyah, self.frame_count, &self.kf);
            }
            track.update_kf(&xyah, &self.kf);
            track.push_observation(xyah, self.frame_count, self.delta_t + 1);
            track.record_match(det.tlwh, det.score, det.class_id);

            if use_appearance {
                track.push_feature(embeddings[*det_idx].clone());
            }
        }

        for det_idx in unmatched_dets {
            let det = &dets[det_idx];
            let feature = use_appearance.then(|| embeddings[det_idx].clone());
            let track = DeepOcSortTrack::new(
                det.tlwh,
                det.score,
                det.class_id,
                self.next_id,
                self.frame_count,
                feature,
                &self.kf,
            );
            self.next_id += 1;
            self.tracks.push(track);
        }

        for track in &mut self.tracks {
            if track.time_since_update == 0 && track.hit_streak >= self.min_hits {
                track.state = TrackState::Confirmed;
            }
            if track.time_since_update > self.max_age {
                track.mark_deleted();
            }
        }

        let unmatched_set: HashSet<usize> = unmatched_trks.into_iter().collect();
        for (i, track) in self.tracks.iter_mut().enumerate() {
            if unmatched_set.contains(&i) && track.state == TrackState::Tentative {
                track.mark_deleted();
            }
        }

        self.tracks.retain(|t| t.state != TrackState::Deleted);

        self.flush_gallery();

        self.tracks
            .iter()
            .filter(|t| t.is_confirmed() && t.time_since_update == 0)
            .cloned()
            .collect()
    }

    /// Push each track's buffered embeddings into the gallery and drop galleries
    /// for tracks that no longer exist.
    fn flush_gallery(&mut self) {
        let active: Vec<u64> = self.tracks.iter().map(|t| t.track_id).collect();
        let mut features: Vec<(u64, Vec<f32>)> = Vec::new();
        for track in &mut self.tracks {
            let id = track.track_id;
            for feature in track.take_features() {
                features.push((id, feature));
            }
        }
        if !features.is_empty() || !active.is_empty() {
            self.metric.partial_fit(&features, &active);
        }
    }

    /// Two-round association: motion (IoU + OCM) blended with appearance in round 1,
    /// then a motion-only round 2 on last observed positions.
    ///
    /// Returns matches as `(detection, track)` pairs plus the unmatched detections
    /// and tracks.
    fn associate(
        &self,
        detections: &[Detection],
        embeddings: &[Vec<f32>],
        use_appearance: bool,
    ) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
        let n_trks = self.tracks.len();
        let n_dets = detections.len();

        if n_trks == 0 {
            return (Vec::new(), (0..n_dets).collect(), Vec::new());
        }
        if n_dets == 0 {
            return (Vec::new(), Vec::new(), (0..n_trks).collect());
        }

        let pred_boxes: Vec<[f32; 4]> = self.tracks.iter().map(|t| t.tlwh).collect();
        let det_boxes: Vec<[f32; 4]> = detections.iter().map(|d| d.tlwh).collect();
        let det_scores: Vec<f32> = detections.iter().map(|d| d.score).collect();

        let ious = iou_batch(&pred_boxes, &det_boxes);
        let angle_diff = ocm_angle_bonus(
            &self.tracks,
            &det_boxes,
            &det_scores,
            self.delta_t,
            self.inertia,
        );

        // Appearance cost matrix (n_trks x n_dets) of cosine distances, or empty.
        let app_cost = if use_appearance {
            let track_ids: Vec<u64> = self.tracks.iter().map(|t| t.track_id).collect();
            self.metric.distance(embeddings, &track_ids)
        } else {
            Vec::new()
        };

        // Round 1: motion (IoU + OCM) blended with the gated appearance cost.
        let cost_matrix: Vec<Vec<f32>> = (0..n_trks)
            .map(|i| {
                (0..n_dets)
                    .map(|j| {
                        let motion_cost = 1.0 - (ious[i][j] + angle_diff[i][j]);
                        if use_appearance {
                            let app = app_cost[i][j];
                            if app <= self.max_cosine_distance {
                                let eff_w = self.appearance_weight * detections[j].score;
                                return (1.0 - eff_w) * motion_cost + eff_w * (app / 2.0);
                            }
                        }
                        motion_cost
                    })
                    .collect()
            })
            .collect();

        let (matches_raw, mut unmatched_trks, mut unmatched_dets) =
            greedy_match(&cost_matrix, 1.0 - self.iou_threshold);
        let mut matches: Vec<(usize, usize)> = matches_raw
            .into_iter()
            .map(|(trk, det)| (det, trk))
            .collect();

        // Round 2: motion-only re-match on last observed positions.
        last_observation_rematch(
            &self.tracks,
            &det_boxes,
            &mut matches,
            &mut unmatched_dets,
            &mut unmatched_trks,
            self.iou_threshold,
        );

        (matches, unmatched_dets, unmatched_trks)
    }
}
