//! Shared two-stage association cascade for the ByteTrack family.
//!
//! ByteTrack and BoT-SORT run the same cascade. High confidence detections are matched
//! against the pool of tracked and lost tracks, low confidence detections then try to
//! recover the still-tracked leftovers, unmatched high detections start new tracks, and
//! originally-lost tracks are kept alive until they age out of the buffer.
//!
//! The only thing that differs between the two is the stage-one cost matrix. ByteTrack
//! passes a plain IoU distance; BoT-SORT passes an appearance-fused cost. Camera motion
//! is applied by BoT-SORT before the cascade runs. Everything else lives here once.

use crate::utils::assignment::{greedy_match, iou_match};
use crate::utils::kalman::KalmanFilter;

/// A detection the cascade can read: its box and its score.
pub(crate) trait CascadeDet {
    /// The detection box in TLWH form.
    fn det_box(&self) -> [f32; 4];
    /// The detection confidence.
    fn det_score(&self) -> f32;
}

/// A track the cascade can drive through the ByteTrack lifecycle.
pub(crate) trait CascadeTrack: Sized + Clone {
    /// The detection type this track is matched against.
    type Det: CascadeDet;

    /// Whether the track is currently in the tracked state.
    fn is_tracked(&self) -> bool;
    /// Whether the track is currently in the lost state.
    fn is_lost(&self) -> bool;
    /// Move the track into the lost state.
    fn set_lost(&mut self);
    /// Frame id of the track's most recent match.
    fn cascade_frame_id(&self) -> usize;
    /// Current box estimate in TLWH form.
    fn cascade_tlwh(&self) -> [f32; 4];

    /// Continue an already-tracked track with a matched detection.
    fn update_matched(&mut self, det: &Self::Det, frame_id: usize, kf: &KalmanFilter);
    /// Bring a lost track back with a matched detection, keeping its id.
    fn reactivate_matched(&mut self, det: &Self::Det, frame_id: usize, kf: &KalmanFilter);
    /// Create and activate a brand-new track from a detection.
    fn spawn(det: &Self::Det, frame_id: usize, track_id: u64, kf: &KalmanFilter) -> Self;
}

/// The tracks produced by one cascade pass, ready for the caller to commit.
pub(crate) struct CascadeOutcome<T> {
    /// Tracks matched this frame or freshly spawned.
    pub activated: Vec<T>,
    /// Lost tracks recovered this frame.
    pub refind: Vec<T>,
    /// Tracks that are lost and still within the buffer.
    pub lost: Vec<T>,
}

/// Run the two-stage cascade over a pool of tracks.
///
/// `pool` holds the tracked tracks first (`0..n_tracked`) then the lost tracks.
/// `stage1_cost[i][j]` is the cost of pairing pool track `i` with high detection `j`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run<T: CascadeTrack>(
    mut pool: Vec<T>,
    n_tracked: usize,
    high: &[T::Det],
    low: &[T::Det],
    stage1_cost: &[Vec<f32>],
    match_thresh: f32,
    second_match_thresh: f32,
    det_thresh: f32,
    buffer_size: usize,
    frame_id: usize,
    kf: &KalmanFilter,
    next_id: &mut u64,
) -> CascadeOutcome<T> {
    let mut activated = Vec::new();
    let mut refind = Vec::new();
    let mut lost = Vec::new();

    // Stage 1: high-confidence detections against the whole pool. Guard the empty case
    // because greedy_match on an empty matrix cannot report the unmatched detections.
    let (matches, u_track, u_det_high) = if pool.is_empty() || high.is_empty() {
        (
            Vec::new(),
            (0..pool.len()).collect::<Vec<_>>(),
            (0..high.len()).collect::<Vec<_>>(),
        )
    } else {
        greedy_match(stage1_cost, match_thresh)
    };

    for (itrack, idet) in matches {
        if pool[itrack].is_tracked() {
            pool[itrack].update_matched(&high[idet], frame_id, kf);
            activated.push(pool[itrack].clone());
        } else {
            pool[itrack].reactivate_matched(&high[idet], frame_id, kf);
            refind.push(pool[itrack].clone());
        }
    }

    // Stage 2: low-confidence detections against still-tracked leftovers, IoU only.
    let r_tracked: Vec<usize> = u_track
        .iter()
        .copied()
        .filter(|&i| pool[i].is_tracked())
        .collect();
    let r_boxes: Vec<[f32; 4]> = r_tracked.iter().map(|&i| pool[i].cascade_tlwh()).collect();
    let low_boxes: Vec<[f32; 4]> = low.iter().map(|d| d.det_box()).collect();
    let (matches_low, u_track_low, _) = iou_match(&r_boxes, &low_boxes, second_match_thresh);

    for (local, idet) in matches_low {
        let itrack = r_tracked[local];
        pool[itrack].update_matched(&low[idet], frame_id, kf);
        activated.push(pool[itrack].clone());
    }

    // Tracked leftovers from the second stage become lost.
    for &local in &u_track_low {
        let itrack = r_tracked[local];
        if !pool[itrack].is_lost() {
            pool[itrack].set_lost();
            lost.push(pool[itrack].clone());
        }
    }

    // Unmatched high detections above the init threshold start new tracks.
    for &idet in &u_det_high {
        if high[idet].det_score() < det_thresh {
            continue;
        }
        let track = T::spawn(&high[idet], frame_id, *next_id, kf);
        *next_id += 1;
        activated.push(track);
    }

    // Keep originally-lost tracks that stayed unmatched alive until they exceed the
    // buffer. Tracks that just became lost in stage two are already in `lost`, so this
    // is restricted to the originally-lost part of the pool (indices at or past
    // `n_tracked`).
    for &i in &u_track {
        if i >= n_tracked && frame_id - pool[i].cascade_frame_id() <= buffer_size {
            lost.push(pool[i].clone());
        }
    }

    CascadeOutcome {
        activated,
        refind,
        lost,
    }
}
