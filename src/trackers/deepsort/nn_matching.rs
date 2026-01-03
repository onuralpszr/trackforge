use std::collections::HashMap;

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
            if let Some(b) = self.budget {
                if sample_list.len() > b {
                    // Remove oldest (simple FIFO)
                    let remove_count = sample_list.len() - b;
                    sample_list.drain(0..remove_count);
                }
            }
        }

        // Filter out inactive targets
        // Creating a set for faster lookup might be better if many targets, but slice is fine for now.
        self.samples.retain(|k, _| active_targets.contains(k));
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
                Metric::Cosine => cosine_distance(sample, feature),
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

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    // 1 - (a . b) / (|a| * |b|)
    // Assuming normalized features, it's just 1 - a . b
    // But let's be safe and compute norms.
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    let cosine_sim = if norm_a > 1e-6 && norm_b > 1e-6 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    };

    // Clamp to [0, 1] for safety in case of float errors, though dot/norms should be <= 1.
    // Distance is 1 - similarity.
    (1.0 - cosine_sim).max(0.0)
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
    fn test_cosine() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        // Orthogonal -> dist 1.0
        assert!((cosine_distance(&a, &b) - 1.0).abs() < 1e-5);

        let c = vec![1.0, 0.0];
        // Same -> dist 0.0
        assert!((cosine_distance(&a, &c)).abs() < 1e-5);
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
}
