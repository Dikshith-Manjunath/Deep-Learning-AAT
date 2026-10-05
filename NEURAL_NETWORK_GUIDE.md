# 2-Layer Neural Network: Financial Credit Risk & Loan Default Classifier

A high-performance, mathematically rigorous 2-layer Artificial Neural Network (Multilayer Perceptron) implemented from scratch in **Rust** with hardware-level optimizations (cache-locality, zero heap allocation, SIMD autovectorization).

---

## 1. Network Architecture & Propagation Diagram

Below is the complete visual diagram showing the **nodes**, **synaptic weights ($W$)**, **balances / biases ($b$)**, **forward data propagation**, and **backpropagation error gradients**:

![2-Layer Neural Network Architecture and Data Propagation](file:///home/dikshith/Documents/SEM_VII/DL/AAT1/neural_network_diagram.png)

```
[ Input Layer: x ∈ ℝ^(1×24) ]
             │
             │   W₁ (Synaptic Weights: 24×16) + b₁ (Balances / Biases: 1×16)
             ▼
[ Hidden Layer 1: h = ReLU(x·W₁ + b₁) ∈ ℝ^(1×16) ]
             │
             │   W₂ (Synaptic Weights: 16×1) + b₂ (Balances / Biases: 1×1)
             ▼
[ Output Layer: ŷ = Sigmoid(h·W₂ + b₂) ∈ [0, 1] (Probability of Default) ]
```

---

## 2. Dataset: German Credit Risk / Loan Default Benchmark

- **Source**: UCI Machine Learning Repository (*Statlog German Credit Data*).
- **Location**: [`data/credit_risk.csv`](file:///home/dikshith/Documents/SEM_VII/DL/AAT1/data/credit_risk.csv)
- **Samples**: $1,000$ total ($800$ Training, $200$ Unseen Test).
- **Features ($24$ continuous / normalized indicators)**:
  - Account checking status, loan duration (months), credit history score, loan purpose, credit amount, savings tier, employment duration, installment rate, personal debtor status, age, collateral, housing type, existing credits, job category, etc.
- **Target Variable**:
  - `0`: **Low Risk / Good Credit (Loan Approved)**
  - `1`: **High Risk / Bad Credit (Loan Default)**
- **Preprocessing**: Feature-wise Z-Score Standardization ($\mu = 0, \sigma = 1$).

---

## 3. Theoretical Foundations: Weights, Balances (Biases), & Nodes

### 3.1 The Concept of Nodes (Neurons)
Each node carries out two fundamental mathematical operations:
1. **Affine Combination (Integration)**: Takes the weighted sum of inputs and shifts it by a balance offset (*bias*):
   $$z_j = \sum_{i=1}^{n} x_i W_{ij} + b_j$$
2. **Non-Linear Activation**:
   $$a_j = \sigma(z_j)$$

### 3.2 What are Weights ($W$)?
- **Synaptic Connection Strengths**: Determines the slope, orientation, and relative importance of each feature in the multi-dimensional decision hyperplanes.
- **Matrix Dimensions**:
  - $\mathbf{W_1} \in \mathbb{R}^{24 \times 16}$ ($384$ learnable parameters)
  - $\mathbf{W_2} \in \mathbb{R}^{16 \times 1}$ ($16$ learnable parameters)

### 3.3 What are Balances / Biases ($b$)?
- **Threshold Offsets**: Without a bias term ($b$), all decision boundaries would be forced to pass through the origin $(0, 0, \dots, 0)$.
- **Role**: Shifts the activation function horizontally along the axis ($x W = -b$). If all incoming features are zero, the balance parameter determines the baseline probability of default.
- **Vector Dimensions**:
  - $\mathbf{b_1} \in \mathbb{R}^{1 \times 16}$
  - $\mathbf{b_2} \in \mathbb{R}^{1 \times 1}$

---

## 4. How to Run the Demo & Obtain Outputs

### 4.1 Quick Run (Build & Execute in Release Mode)
In your terminal, simply execute:

```bash
cargo run --release
```

### 4.2 What the Terminal Output Displays:
1. **Data Preprocessing Summary**: Confirms dataset loading, feature count ($24$), and $80/20$ train-test split.
2. **Training Progression**: Epoch-by-epoch loss, training accuracy, and training F1-score.
3. **Comprehensive Test Dashboard**:
   - **Overall Accuracy**
   - **Precision (Positives / Default)**
   - **Recall / Sensitivity**
   - **F1-Score**
   - **Average Confidence Score**
   - **Full Confusion Matrix** ($TP, FP, TN, FN$)
4. **Live Inference Demo**: Individual risk predictions for sample test applicants showing:
   - **Predicted Risk Probability** ($P(\text{Default})$)
   - **Model Decision**: `[LOW RISK (APPROVED)]` vs `[HIGH RISK (DEFAULT)]`
   - **Confidence Score**: e.g., `100.00%` or `72.26%`
   - **Ground Truth & Verification Badge** (`✓ CORRECT`)

---

## 5. Rust Code Structure

- [`src/main.rs`](file:///home/dikshith/Documents/SEM_VII/DL/AAT1/src/main.rs): Main pipeline coordinator for data loading, training, test evaluation, and live applicant scoring.
- [`src/dataset.rs`](file:///home/dikshith/Documents/SEM_VII/DL/AAT1/src/dataset.rs): High-speed CSV reader, Z-score standardizer, and train/test partitioner.
- [`src/metrics.rs`](file:///home/dikshith/Documents/SEM_VII/DL/AAT1/src/metrics.rs): Evaluator computing Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and Confidence metrics.
- [`src/nn.rs`](file:///home/dikshith/Documents/SEM_VII/DL/AAT1/src/nn.rs): The 2-Layer Neural Network struct with pre-allocated zero-allocation tensor caches and SGD with Momentum.
- [`src/matrix.rs`](file:///home/dikshith/Documents/SEM_VII/DL/AAT1/src/matrix.rs): Cache-friendly 1D contiguous row-major matrix multiplication and linear algebra engine.
- [`src/activation.rs`](file:///home/dikshith/Documents/SEM_VII/DL/AAT1/src/activation.rs): Vectorized activation functions (ReLU, Sigmoid, Tanh, Linear) and exact analytical derivatives.
