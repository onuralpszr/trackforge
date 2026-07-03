use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    Euclidean,
    Cosine,
}

/// A nearest neighbor distance metric for deep association.
///
/// Keeps a history of features (samples) for each target (track) and computes
/// the minimum distance between a new feature and the stored history.
#[derive(Debug, Clone)]
pub struct NearestNeighborDistanceMetric {
    metric: Metric,
    matching_threshold: f32,
    budget: Option<usize>,
    /// Map from track_id to a list of feature vectors.
    samples: HashMap<u64, Vec<Vec<f32>>>,
}

impl NearestNeighborDistanceMetric {
    /// Create a new distance metric.
    ///
    /// # Arguments
    /// * `metric` - The distance metric to use (Euclidean or Cosine).
    /// * `matching_threshold` - Threshold for matching.
    /// * `budget` - Optional maximum number of samples to keep per track.
    pub fn new(metric: Metric, matching_threshold: f32, budget: Option<usize>) -> Self {
        Self {
            metric,
            matching_threshold,
            budget,
            samples: HashMap::new(),
        }
    }

    /// Update the sample gallery with new features.
    ///
    /// # Arguments
    /// * `features` - Map from track_id to a list of new features.
    /// * `active_targets` - List of track IDs that are currently active (confirmed).
    ///   Sample galleries for inactive targets will be removed.
    pub fn partial_fit(&mut self, features: &[(u64, Vec<f32>)], active_targets: &[u64]) {
        for (track_id, feature) in features {
            let sample_list = self.samples.entry(*track_id).or_default();
            sample_list.push(feature.clone());
            if let Some(b) = self.budget
                && sample_list.len() > b
            {
                let remove_count = sample_list.len() - b;
                sample_list.drain(0..remove_count);
            }
        }

        let active_set: HashSet<u64> = active_targets.iter().cloned().collect();
        self.samples.retain(|k, _| active_set.contains(k));
    }

    /// Compute the distance matrix between tracks and detections.
    ///
    /// # Arguments
    /// * `features` - A map of detection indices to their feature vectors
    ///   (usually we pass a list of features corresponding to detections).
    /// * `targets` - List of track IDs to compare against.
    ///
    /// # Returns
    /// An n_targets x n_features matrix of distances.
    pub fn distance(&self, features: &[Vec<f32>], targets: &[u64]) -> Vec<Vec<f32>> {
        let mut cost_matrix = vec![vec![0.0; features.len()]; targets.len()];

        for (i, track_id) in targets.iter().enumerate() {
            let sample_list = match self.samples.get(track_id) {
                Some(s) => s,
                None => {
                    // keeping 0.0 or MAX? If no samples, distance is undefined or max.
                    // Usually tracks passed here should have samples.
                    // Let's set to max distance to avoid matching.
                    for cell in cost_matrix[i].iter_mut() {
                        *cell = f32::MAX;
                    }
                    continue;
                }
            };

            for (j, feature) in features.iter().enumerate() {
                cost_matrix[i][j] = self.compute_min_distance(sample_list, feature);
            }
        }

        cost_matrix
    }

    fn compute_min_distance(&self, samples: &[Vec<f32>], feature: &[f32]) -> f32 {
        let mut min_dist = f32::MAX;
        for sample in samples {
            let dist = match self.metric {
                Metric::Euclidean => euclidean_distance(sample, feature),
                Metric::Cosine => crate::utils::features::cosine_distance(sample, feature),
            };
            if dist < min_dist {
                min_dist = dist;
            }
        }
        min_dist
    }

    pub fn matching_threshold(&self) -> f32 {
        self.matching_threshold
    }
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        sum += (x - y).powi(2);
    }
    sum.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((euclidean_distance(&a, &b) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_metric_budget() {
        let mut metric = NearestNeighborDistanceMetric::new(Metric::Euclidean, 0.5, Some(2));

        // Add 3 samples for track 1
        metric.partial_fit(&[(1, vec![1.0]), (1, vec![2.0]), (1, vec![3.0])], &[1]);

        let samples = metric.samples.get(&1).unwrap();
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0], vec![2.0]);
        assert_eq!(samples[1], vec![3.0]);
    }

    #[test]
    fn test_metric_no_budget() {
        // Test without budget limit
        let mut metric = NearestNeighborDistanceMetric::new(Metric::Cosine, 0.3, None);

        // Add many samples
        metric.partial_fit(
            &[
                (1, vec![1.0, 0.0]),
                (1, vec![0.9, 0.1]),
                (1, vec![0.8, 0.2]),
                (1, vec![0.7, 0.3]),
            ],
            &[1],
        );

        let samples = metric.samples.get(&1).unwrap();
        assert_eq!(samples.len(), 4); // All samples kept
    }

    #[test]
    fn test_metric_inactive_removal() {
        let mut metric = NearestNeighborDistanceMetric::new(Metric::Euclidean, 0.5, Some(10));

        // Add samples for tracks 1 and 2
        metric.partial_fit(&[(1, vec![1.0]), (2, vec![2.0])], &[1, 2]);
        assert!(metric.samples.contains_key(&1));
        assert!(metric.samples.contains_key(&2));

        // Now only track 1 is active - track 2 should be removed
        metric.partial_fit(&[(1, vec![1.5])], &[1]);
        assert!(metric.samples.contains_key(&1));
        assert!(!metric.samples.contains_key(&2));
    }

    #[test]
    fn test_distance_matrix() {
        let mut metric = NearestNeighborDistanceMetric::new(Metric::Euclidean, 0.5, Some(10));

        // Add samples for track 1
        metric.partial_fit(&[(1, vec![0.0, 0.0])], &[1]);

        // Compute distances to new features
        let features = vec![vec![0.0, 0.0], vec![3.0, 4.0]]; // distances: 0 and 5
        let cost_matrix = metric.distance(&features, &[1]);

        assert_eq!(cost_matrix.len(), 1); // 1 target
        assert_eq!(cost_matrix[0].len(), 2); // 2 features
        assert!(cost_matrix[0][0] < 0.01); // Same point
        assert!((cost_matrix[0][1] - 5.0).abs() < 0.01); // (3,4) distance
    }

    #[test]
    fn test_distance_no_samples() {
        let metric = NearestNeighborDistanceMetric::new(Metric::Euclidean, 0.5, Some(10));

        // Track 1 has no samples
        let features = vec![vec![0.0, 0.0]];
        let cost_matrix = metric.distance(&features, &[1]);

        // Should return MAX distance for unknown track
        assert_eq!(cost_matrix.len(), 1);
        assert_eq!(cost_matrix[0][0], f32::MAX);
    }

    #[test]
    fn test_matching_threshold() {
        let metric = NearestNeighborDistanceMetric::new(Metric::Cosine, 0.25, Some(10));
        assert!((metric.matching_threshold() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_min_distance_multiple_samples() {
        let mut metric = NearestNeighborDistanceMetric::new(Metric::Euclidean, 0.5, Some(10));

        // Add multiple samples for track 1
        metric.partial_fit(
            &[
                (1, vec![0.0, 0.0]),
                (1, vec![10.0, 0.0]),
                (1, vec![5.0, 0.0]),
            ],
            &[1],
        );

        // Feature at (1, 0) - closest to (0, 0)
        let features = vec![vec![1.0, 0.0]];
        let cost_matrix = metric.distance(&features, &[1]);

        // Min distance should be 1.0 (to origin sample)
        assert!((cost_matrix[0][0] - 1.0).abs() < 0.01);
    }
}
