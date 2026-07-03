//! Appearance-feature helpers shared by the Re-ID trackers.
//!
//! DeepSORT's feature gallery and BoT-SORT's per-track embedding both need the same
//! two operations: L2-normalise a vector and measure the cosine distance between two
//! vectors. They live here so there is one tested implementation.

/// L2-normalise a feature vector.
///
/// A zero (or near-zero) vector is returned unchanged, so callers never divide by zero.
pub fn l2_normalize(feature: &[f32]) -> Vec<f32> {
    let norm = feature.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm <= 1e-12 {
        return feature.to_vec();
    }
    feature.iter().map(|v| v / norm).collect()
}

/// Cosine distance between two vectors, in `[0, 2]`.
///
/// Each vector is normalised internally, so the inputs need not be unit length.
/// Identical directions give `0`, orthogonal give `1`, opposite give `2`. A
/// zero-norm input yields `1` (no similarity).
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    let cosine_sim = if norm_a > 1e-6 && norm_b > 1e-6 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    };
    (1.0 - cosine_sim).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_normalize_makes_unit_vectors() {
        let unit = l2_normalize(&[3.0, 4.0]);
        assert!((unit[0] - 0.6).abs() < 1e-6 && (unit[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_leaves_zero_vector() {
        assert_eq!(l2_normalize(&[0.0, 0.0, 0.0]), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn cosine_orthogonal_and_identical() {
        assert!((cosine_distance(&[1.0, 0.0], &[0.0, 1.0]) - 1.0).abs() < 1e-5);
        assert!(cosine_distance(&[1.0, 0.0], &[1.0, 0.0]).abs() < 1e-5);
    }

    #[test]
    fn cosine_ignores_magnitude() {
        // Same direction, different magnitude -> distance ~0.
        assert!(cosine_distance(&[1.0, 1.0], &[2.0, 2.0]) < 0.01);
    }

    #[test]
    fn cosine_opposite_is_two() {
        assert!((cosine_distance(&[1.0, 0.0], &[-1.0, 0.0]) - 2.0).abs() < 0.01);
    }

    #[test]
    fn cosine_zero_norm_is_one() {
        assert!((cosine_distance(&[0.0, 0.0], &[1.0, 1.0]) - 1.0).abs() < 0.01);
    }
}
