use crate::matrix::Matrix;

#[allow(dead_code)]
pub struct ClassificationMetrics {

    pub total: usize,
    pub true_positives: usize,
    pub false_positives: usize,
    pub true_negatives: usize,
    pub false_negatives: usize,
    pub accuracy: f32,
    pub precision: f32,
    pub recall: f32,
    pub f1_score: f32,
    pub avg_confidence: f32,
}

impl ClassificationMetrics {
    pub fn evaluate(predictions: &Matrix, targets: &Matrix, threshold: f32) -> Self {
        assert_eq!(predictions.data.len(), targets.data.len());
        let total = predictions.data.len();

        let mut tp = 0;
        let mut fp = 0;
        let mut tn = 0;
        let mut fn_count = 0;
        let mut conf_sum = 0.0;

        for (pred, target) in predictions.data.iter().zip(&targets.data) {
            let actual = *target >= 0.5;
            let predicted = *pred >= threshold;
            let prob = *pred;

            // Confidence is how certain the model is about its chosen decision
            let confidence = if predicted { prob } else { 1.0 - prob };
            conf_sum += confidence;

            match (predicted, actual) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, false) => tn += 1,
                (false, true) => fn_count += 1,
            }
        }

        let accuracy = (tp + tn) as f32 / total as f32;
        let precision = if tp + fp > 0 {
            tp as f32 / (tp + fp) as f32
        } else {
            0.0
        };
        let recall = if tp + fn_count > 0 {
            tp as f32 / (tp + fn_count) as f32
        } else {
            0.0
        };
        let f1_score = if precision + recall > 0.0 {
            2.0 * (precision * recall) / (precision + recall)
        } else {
            0.0
        };
        let avg_confidence = conf_sum / total as f32;

        Self {
            total,
            true_positives: tp,
            false_positives: fp,
            true_negatives: tn,
            false_negatives: fn_count,
            accuracy,
            precision,
            recall,
            f1_score,
            avg_confidence,
        }
    }

    pub fn print_dashboard(&self, title: &str) {
        println!("\n╔════════════════════════════════════════════════════════════════╗");
        println!("║ {:^62} ║", title);
        println!("╠════════════════════════════════════════════════════════════════╣");
        println!(
            "║  Overall Accuracy:      {:>6.2}%                               ║",
            self.accuracy * 100.0
        );
        println!(
            "║  Precision (Positives): {:>6.2}%                               ║",
            self.precision * 100.0
        );
        println!(
            "║  Recall / Sensitivity:  {:>6.2}%                               ║",
            self.recall * 100.0
        );
        println!(
            "║  F1-Score:              {:>6.4}                                ║",
            self.f1_score
        );
        println!(
            "║  Average Confidence:    {:>6.2}%                               ║",
            self.avg_confidence * 100.0
        );
        println!("╠════════════════════════════════════════════════════════════════╣");
        println!("║                      CONFUSION MATRIX                          ║");
        println!("║                                                                ║");
        println!("║                    Predicted Positive     Predicted Negative   ║");
        println!(
            "║  Actual Positive:     {:>6} (TP)               {:>6} (FN)     ║",
            self.true_positives, self.false_negatives
        );
        println!(
            "║  Actual Negative:     {:>6} (FP)               {:>6} (TN)     ║",
            self.false_positives, self.true_negatives
        );
        println!("╚════════════════════════════════════════════════════════════════╝");
    }
}
