use nalgebra::{SMatrix, SVector};

pub type StateVector = SVector<f32, 8>; // [x, y, a, h, vx, vy, va, vh]
pub type MeasurementVector = SVector<f32, 4>; // [x, y, a, h]
pub type CovarianceMatrix = SMatrix<f32, 8, 8>;
pub type MeasurementMatrix = SMatrix<f32, 4, 8>;

#[derive(Debug, Clone)]
pub struct KalmanFilter {
    motion_mat: SMatrix<f32, 8, 8>,
    update_mat: MeasurementMatrix,
    std_weight_position: f32,
    std_weight_velocity: f32,
}

impl KalmanFilter {
    pub fn new(std_weight_position: f32, std_weight_velocity: f32) -> Self {
        let mut motion_mat = SMatrix::<f32, 8, 8>::identity();
        for i in 0..4 {
            motion_mat[(i, i + 4)] = 1.0;
        }

        let mut update_mat = MeasurementMatrix::zeros();
        for i in 0..4 {
            update_mat[(i, i)] = 1.0;
        }

        Self {
            motion_mat,
            update_mat,
            std_weight_position,
            std_weight_velocity,
        }
    }

    pub fn initiate(&self, measurement: &MeasurementVector) -> (StateVector, CovarianceMatrix) {
        let mut mean = StateVector::zeros();
        for i in 0..4 {
            mean[i] = measurement[i];
        }

        let mut covariance = CovarianceMatrix::identity();
        let std = [
            2.0 * self.std_weight_position * measurement[3],
            2.0 * self.std_weight_position * measurement[3],
            1e-2,
            2.0 * self.std_weight_position * measurement[3],
            10.0 * self.std_weight_velocity * measurement[3],
            10.0 * self.std_weight_velocity * measurement[3],
            1e-5,
            10.0 * self.std_weight_velocity * measurement[3],
        ];

        for i in 0..8 {
            covariance[(i, i)] = std[i].powi(2);
        }

        (mean, covariance)
    }

    pub fn predict(
        &self,
        mean: &StateVector,
        covariance: &CovarianceMatrix,
    ) -> (StateVector, CovarianceMatrix) {
        let std_pos = [
            self.std_weight_position * mean[3],
            self.std_weight_position * mean[3],
            1e-2,
            self.std_weight_position * mean[3],
        ];
        let std_vel = [
            self.std_weight_velocity * mean[3],
            self.std_weight_velocity * mean[3],
            1e-5,
            self.std_weight_velocity * mean[3],
        ];

        let mut motion_cov = CovarianceMatrix::zeros();
        for i in 0..4 {
            motion_cov[(i, i)] = std_pos[i].powi(2);
            motion_cov[(i + 4, i + 4)] = std_vel[i].powi(2);
        }

        let mean = self.motion_mat * mean;
        let covariance = self.motion_mat * covariance * self.motion_mat.transpose() + motion_cov;

        (mean, covariance)
    }

    pub fn update(
        &self,
        mean: &StateVector,
        covariance: &CovarianceMatrix,
        measurement: &MeasurementVector,
    ) -> (StateVector, CovarianceMatrix) {
        let projected_mean = self.update_mat * mean;
        let projected_cov = self.update_mat * covariance * self.update_mat.transpose();

        let std = [
            self.std_weight_position * mean[3],
            self.std_weight_position * mean[3],
            1e-1,
            self.std_weight_position * mean[3],
        ];
        let mut diag = SMatrix::<f32, 4, 4>::zeros();
        for i in 0..4 {
            diag[(i, i)] = std[i].powi(2);
        }

        let innovation_cov = projected_cov + diag;
        // let inv_innovation_cov = innovation_cov.try_inverse().unwrap(); // Handle unwrap properly in prod
        // Simplification for stability - often solved via Cholesky decomposition or similar
        // For now, assume invertibility for this standard KF setup.
        let inv_innovation_cov = innovation_cov
            .try_inverse()
            .unwrap_or_else(|| SMatrix::<f32, 4, 4>::identity());

        let kalman_gain = covariance * self.update_mat.transpose() * inv_innovation_cov;
        let innovation = measurement - projected_mean;

        let new_mean = mean + kalman_gain * innovation;
        let new_covariance = covariance - kalman_gain * innovation_cov * kalman_gain.transpose();

        (new_mean, new_covariance)
    }
}

impl Default for KalmanFilter {
    fn default() -> Self {
        Self::new(1.0 / 20.0, 1.0 / 160.0)
    }
}
