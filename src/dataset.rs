use crate::matrix::Matrix;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[allow(dead_code)]
pub struct Dataset {

    pub train_x: Matrix,
    pub train_y: Matrix,
    pub test_x: Matrix,
    pub test_y: Matrix,
    pub means: Vec<f32>,
    pub stds: Vec<f32>,
}

#[allow(dead_code)]
impl Dataset {

    /// Loads a CSV file where the last column is the binary target (0 or 1)
    pub fn load_from_csv(path: &str, train_ratio: f32) -> Result<Self, std::io::Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut all_features: Vec<Vec<f32>> = Vec::new();
        let mut all_targets: Vec<f32> = Vec::new();

        for line in reader.lines() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let parts: Vec<&str> = trimmed.split(',').map(|s| s.trim()).collect();
            if parts.len() < 2 {
                continue;
            }

            let n_cols = parts.len();
            let mut row_feats = Vec::with_capacity(n_cols - 1);

            for val in &parts[0..n_cols - 1] {
                if let Ok(num) = val.parse::<f32>() {
                    row_feats.push(num);
                } else {
                    row_feats.push(0.0);
                }
            }

            let target: f32 = parts[n_cols - 1].parse().unwrap_or(0.0);

            all_features.push(row_feats);
            all_targets.push(target);
        }

        let num_samples = all_features.len();
        assert!(num_samples > 0, "Dataset is empty");
        let num_features = all_features[0].len();

        // Compute column means and standard deviations for Z-score normalization
        let mut means = vec![0.0f32; num_features];
        let mut stds = vec![0.0f32; num_features];

        for row in &all_features {
            for j in 0..num_features {
                means[j] += row[j];
            }
        }
        for j in 0..num_features {
            means[j] /= num_samples as f32;
        }

        for row in &all_features {
            for j in 0..num_features {
                let diff = row[j] - means[j];
                stds[j] += diff * diff;
            }
        }
        for j in 0..num_features {
            stds[j] = (stds[j] / (num_samples as f32).max(1.0)).sqrt();
            if stds[j] < 1e-6 {
                stds[j] = 1.0; // avoid division by zero for constant features
            }
        }

        // Normalize features
        let mut normalized_data = Vec::with_capacity(num_samples * num_features);
        for row in &all_features {
            for j in 0..num_features {
                let norm = (row[j] - means[j]) / stds[j];
                normalized_data.push(norm);
            }
        }

        // Deterministic split: first train_count for training, rest for test
        let train_count = ((num_samples as f32) * train_ratio).round() as usize;
        let test_count = num_samples - train_count;

        let train_x_data = normalized_data[0..train_count * num_features].to_vec();
        let train_y_data = all_targets[0..train_count].to_vec();

        let test_x_data = normalized_data[train_count * num_features..].to_vec();
        let test_y_data = all_targets[train_count..].to_vec();

        let train_x = Matrix::from_vec(train_count, num_features, train_x_data);
        let train_y = Matrix::from_vec(train_count, 1, train_y_data);
        let test_x = Matrix::from_vec(test_count, num_features, test_x_data);
        let test_y = Matrix::from_vec(test_count, 1, test_y_data);

        Ok(Self {
            train_x,
            train_y,
            test_x,
            test_y,
            means,
            stds,
        })
    }

    /// Normalizes a single raw sample using the dataset's fitted parameters
    pub fn normalize_sample(&self, raw: &[f32]) -> Vec<f32> {
        raw.iter()
            .enumerate()
            .map(|(i, &val)| {
                let m = self.means.get(i).copied().unwrap_or(0.0);
                let s = self.stds.get(i).copied().unwrap_or(1.0);
                (val - m) / s
            })
            .collect()
    }
}
