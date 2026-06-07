//! Greedy linear assignment shared by the trackers.
//!
//! All four trackers resolve a cost matrix into matches the same way: enumerate
//! every (row, column) cost, sort ascending, and greedily accept a pair when both
//! its row and column are still free and the cost does not exceed the threshold.

use std::collections::HashSet;

/// Greedily match rows to columns of a cost matrix.
///
/// `cost_matrix[r][c]` is the cost of pairing row `r` with column `c`. Pairs are
/// accepted in ascending cost order while both endpoints are unmatched and the
/// cost is `<= threshold`.
///
/// Returns `(matches, unmatched_rows, unmatched_cols)` where each match is a
/// `(row, col)` pair. The unmatched vectors are unordered.
pub fn greedy_match(
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

    let mut costs: Vec<(f32, usize, usize)> = Vec::with_capacity(rows * cols);
    for (r, row) in cost_matrix.iter().enumerate() {
        for (c, &cost) in row.iter().enumerate() {
            costs.push((cost, r, c));
        }
    }
    costs.sort_by(|a, b| a.0.total_cmp(&b.0));

    for (cost, r, c) in costs {
        if cost > threshold {
            break;
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

    #[test]
    fn empty_matrix_matches_nothing() {
        let (m, ur, uc) = greedy_match(&[], 0.5);
        assert!(m.is_empty());
        assert!(ur.is_empty());
        assert!(uc.is_empty());
    }

    #[test]
    fn picks_lowest_cost_pairs() {
        // Row 0 prefers col 1 (0.1), row 1 prefers col 0 (0.2).
        let cost = vec![vec![0.9, 0.1], vec![0.2, 0.8]];
        let (mut m, ur, uc) = greedy_match(&cost, 0.5);
        m.sort();
        assert_eq!(m, vec![(0, 1), (1, 0)]);
        assert!(ur.is_empty());
        assert!(uc.is_empty());
    }

    #[test]
    fn rejects_costs_above_threshold() {
        let cost = vec![vec![0.9, 0.9], vec![0.9, 0.9]];
        let (m, mut ur, mut uc) = greedy_match(&cost, 0.5);
        ur.sort();
        uc.sort();
        assert!(m.is_empty());
        assert_eq!(ur, vec![0, 1]);
        assert_eq!(uc, vec![0, 1]);
    }

    #[test]
    fn handles_more_rows_than_cols() {
        let cost = vec![vec![0.1], vec![0.2], vec![0.3]];
        let (m, mut ur, uc) = greedy_match(&cost, 1.0);
        assert_eq!(m, vec![(0, 0)]);
        ur.sort();
        assert_eq!(ur, vec![1, 2]);
        assert!(uc.is_empty());
    }
}
