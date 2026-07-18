//! Parameters shared by every tracker.
//!
//! These two settings control the track lifecycle and mean the same thing in every
//! tracker, so they live here once instead of being repeated under different names
//! (`max_age` vs `track_buffer`, `min_hits` vs `n_init`). Each tracker keeps its own
//! specific settings in its own params struct and embeds a [`CommonParams`] for these.

/// Lifecycle settings common to all trackers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommonParams {
    /// How many frames a track stays alive after it stops matching any detection.
    ///
    /// If an object is missed or hidden for up to this many frames, the track keeps
    /// its id and can be picked back up when the object returns. Past this the track
    /// is dropped. Larger values ride out longer occlusions but risk keeping stale
    /// tracks and handing an old id to a different object. Some trackers call this
    /// the track buffer; it is the same thing.
    pub max_age: usize,

    /// How many matched frames in a row a new track needs before it is confirmed and
    /// returned to the caller.
    ///
    /// Larger values suppress flickering false tracks from one-off detections but
    /// delay when a real object first shows up in the output. Some trackers call this
    /// `n_init`; it is the same thing.
    pub min_hits: usize,
}

impl Default for CommonParams {
    fn default() -> Self {
        Self {
            max_age: 30,
            min_hits: 3,
        }
    }
}

impl CommonParams {
    /// Build the common lifecycle settings directly.
    pub fn new(max_age: usize, min_hits: usize) -> Self {
        Self { max_age, min_hits }
    }
}
