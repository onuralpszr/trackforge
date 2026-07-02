//! Observation-centric association helpers shared by OC-SORT and Deep OC-SORT.
//!
//! Both trackers add the same OCM direction-consistency bonus to their IoU cost and
//! run the same round-2 rematch on last observed positions. Only the cost blend in
//! between differs (Deep OC-SORT mixes in appearance), so those two steps live here.

use crate::trackers::common::ObsTrack;
use crate::utils::assignment::greedy_match;
use crate::utils::geometry::{iou_batch, xyah_to_tlwh};

/// OCM direction-consistency bonus matrix (`n_trks x n_dets`).
///
/// For each track with a computable velocity direction, adds a bonus proportional
/// to the cosine similarity between that direction and the direction from the
/// track's last observation to each detection centre, scaled by `inertia` and the
/// detection score. Tracks without a direction contribute a zero row.
pub(crate) fn ocm_angle_bonus(
    tracks: &[ObsTrack],
    det_boxes: &[[f32; 4]],
    det_scores: &[f32],
    delta_t: usize,
    inertia: f32,
) -> Vec<Vec<f32>> {
    let n_dets = det_boxes.len();
    let mut angle_diff = vec![vec![0.0_f32; n_dets]; tracks.len()];

    for (i, track) in tracks.iter().enumerate() {
        let vel_dir = match track.obs_direction(delta_t) {
            Some(v) => v,
            None => continue,
        };
        let last = track.last_observation();
        let (last_cx, last_cy) = (last[0], last[1]);
        for (j, det) in det_boxes.iter().enumerate() {
            let det_cx = det[0] + det[2] / 2.0;
            let det_cy = det[1] + det[3] / 2.0;
            let dy = det_cy - last_cy;
            let dx = det_cx - last_cx;
            let norm = (dy * dy + dx * dx).sqrt() + 1e-6;
            let dot = (vel_dir[0] * (dy / norm) + vel_dir[1] * (dx / norm)).clamp(-1.0, 1.0);
            let angle = dot.acos();
            let normalized = (std::f32::consts::FRAC_PI_2 - angle.abs()) / std::f32::consts::PI;
            angle_diff[i][j] = (normalized * inertia * det_scores[j]).max(0.0);
        }
    }
    angle_diff
}

/// Round-2 rematch on last observed positions.
///
/// For detections and tracks left unmatched by round 1, rematches them by IoU using
/// each track's last *observed* box rather than its Kalman prediction. Appends the
/// new pairs to `matches` (as global `(det, trk)` indices) and shrinks the unmatched
/// lists in place. Does nothing unless some pair reaches `iou_threshold`.
pub(crate) fn last_observation_rematch(
    tracks: &[ObsTrack],
    det_boxes: &[[f32; 4]],
    matches: &mut Vec<(usize, usize)>,
    unmatched_dets: &mut Vec<usize>,
    unmatched_trks: &mut Vec<usize>,
    iou_threshold: f32,
) {
    if unmatched_dets.is_empty() || unmatched_trks.is_empty() {
        return;
    }

    let left_det_boxes: Vec<[f32; 4]> = unmatched_dets.iter().map(|&di| det_boxes[di]).collect();
    let left_trk_obs: Vec<[f32; 4]> = unmatched_trks
        .iter()
        .map(|&ti| xyah_to_tlwh(tracks[ti].last_observation()))
        .collect();

    let iou_left = iou_batch(&left_trk_obs, &left_det_boxes);
    let max_iou = iou_left
        .iter()
        .flat_map(|r| r.iter())
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    if max_iou <= iou_threshold {
        return;
    }

    // Rows are tracks, columns are detections.
    let cost_left: Vec<Vec<f32>> = iou_left
        .iter()
        .map(|row| row.iter().map(|&v| 1.0 - v).collect())
        .collect();
    let (r2_matches, r2_ut, r2_ud) = greedy_match(&cost_left, 1.0 - iou_threshold);

    for (trk_local, det_local) in r2_matches {
        matches.push((unmatched_dets[det_local], unmatched_trks[trk_local]));
    }
    let new_dets: Vec<usize> = r2_ud.into_iter().map(|di| unmatched_dets[di]).collect();
    let new_trks: Vec<usize> = r2_ut.into_iter().map(|ti| unmatched_trks[ti]).collect();
    *unmatched_dets = new_dets;
    *unmatched_trks = new_trks;
}
