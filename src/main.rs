mod activation;
mod dataset;
mod matrix;
mod metrics;
mod nn;

use activation::Activation;
use dataset::Dataset;
use matrix::Matrix;
use metrics::ClassificationMetrics;
use nn::TwoLayerNN;
use std::time::Instant;

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║       2-LAYER NEURAL NETWORK: CREDIT RISK & LOAN DEFAULT          ║");
    println!("║             High-Performance Classification Engine (Rust)         ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");

    // 1. Load and prepare dataset (80% Train, 20% Test)
    let dataset_path = "data/credit_risk.csv";
    println!("\n[1/4] Loading and Normalizing Dataset from '{}'...", dataset_path);
    let ds = match Dataset::load_from_csv(dataset_path, 0.80) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error loading dataset from {}: {}", dataset_path, e);
            return;
        }
    };

    let num_features = ds.train_x.cols;
    let train_samples = ds.train_x.rows;
    let test_samples = ds.test_x.rows;

    println!("      * Features:          {}", num_features);
    println!("      * Training Samples:  {}", train_samples);
    println!("      * Test Set Samples:  {}", test_samples);
    println!("      * Preprocessing:     Z-Score Standard Normalization (mean=0, std=1)");

    // 2. Initialize 2-Layer Neural Network
    // Layer 1: 24 Inputs -> 16 Hidden Neurons (ReLU)
    // Layer 2: 16 Hidden Neurons -> 1 Output Neuron (Sigmoid for Risk Probability)
    let hidden_neurons = 16;
    println!("\n[2/4] Initializing 2-Layer Neural Network Architecture:");
    println!("      * Input Layer:       {} features", num_features);
    println!("      * Synaptic Weights 1: W1 ({} x {})", num_features, hidden_neurons);
    println!("      * Balances / Biases 1: b1 (1 x {})", hidden_neurons);
    println!("      * Hidden Activation: ReLU");
    println!("      * Synaptic Weights 2: W2 ({} x 1)", hidden_neurons);
    println!("      * Balances / Biases 2: b2 (1 x 1)");
    println!("      * Output Activation: Sigmoid (Probability Output P(Default) in [0, 1])");

    let mut model = TwoLayerNN::new(
        num_features,
        hidden_neurons,
        1,
        Activation::ReLU,
        Activation::Sigmoid,
        42,
    );

    // 3. Train the Neural Network
    let epochs = 8_000;
    let learning_rate = 0.05;
    let momentum = 0.90;
    let weight_decay = 1e-4;

    println!("\n[3/4] Training Model ({} epochs, lr={}, momentum={})...", epochs, learning_rate, momentum);
    let start_train = Instant::now();

    for epoch in 1..=epochs {
        let loss = model.train_step(
            &ds.train_x,
            &ds.train_y,
            learning_rate,
            momentum,
            weight_decay,
        );

        if epoch % 2000 == 0 || epoch == 1 {
            // Quick evaluate train accuracy
            let train_preds = model.forward(&ds.train_x);
            let train_metrics = ClassificationMetrics::evaluate(train_preds, &ds.train_y, 0.5);
            println!(
                "      Epoch {:5}/{} | Loss: {:.6} | Train Acc: {:.2}% | Train F1: {:.4}",
                epoch,
                epochs,
                loss,
                train_metrics.accuracy * 100.0,
                train_metrics.f1_score
            );
        }
    }

    let train_duration = start_train.elapsed();
    println!("      ✓ Training Completed in: {:.2?} ({:.1} µs/epoch)", train_duration, train_duration.as_micros() as f64 / epochs as f64);

    // 4. Evaluate on Unseen Test Set
    println!("\n[4/4] Evaluating on Unseen Test Set ({} Applicants)...", test_samples);
    let test_preds = model.forward(&ds.test_x);
    let test_metrics = ClassificationMetrics::evaluate(test_preds, &ds.test_y, 0.5);

    test_metrics.print_dashboard("TEST SET EVALUATION RESULTS (GERMAN CREDIT RISK)");

    // 5. Live Demo: Sample Applicant Risk Scoring & Confidence Prediction
    println!("\n╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║               LIVE INFERENCE DEMO: SAMPLE APPLICANT SCORING               ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");

    let demo_indices = vec![0, 7, 14, 25, 42, 60];
    for (i, &idx) in demo_indices.iter().enumerate() {
        if idx >= test_samples {
            continue;
        }

        // Extract raw features for this sample
        let start_offset = idx * num_features;
        let sample_norm_slice = &ds.test_x.data[start_offset..start_offset + num_features];
        let sample_matrix = Matrix::from_vec(1, num_features, sample_norm_slice.to_vec());

        let pred_prob = model.forward(&sample_matrix).data[0];
        let actual_label = ds.test_y.data[idx];

        let is_high_risk = pred_prob >= 0.5;
        let confidence = if is_high_risk { pred_prob } else { 1.0 - pred_prob } * 100.0;
        let actual_str = if actual_label >= 0.5 { "HIGH RISK (DEFAULT)" } else { "LOW RISK (GOOD CREDIT)" };
        let pred_str = if is_high_risk { "HIGH RISK (DEFAULT)" } else { "LOW RISK (APPROVED)" };

        let status_badge = if (is_high_risk && actual_label >= 0.5) || (!is_high_risk && actual_label < 0.5) {
            "✓ CORRECT"
        } else {
            "✗ MISCLASSIFIED"
        };

        println!("Applicant #{:<2} [Test ID: {:>3}]:", i + 1, idx);
        println!("  ├─ Risk Probability:    {:.4} ({:.2}% chance of default)", pred_prob, pred_prob * 100.0);
        println!("  ├─ Model Decision:      {}", pred_str);
        println!("  ├─ Confidence Score:    {:.2}%", confidence);
        println!("  ├─ Ground Truth:        {}", actual_str);
        println!("  └─ Verification:        {}", status_badge);
        println!("  ─────────────────────────────────────────────────────────────────────────");
    }

    println!("\n✓ Neural Network is ready for your demo presentation!");
}
