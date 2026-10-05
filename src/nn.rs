/// Blazing Fast 2-Layer Neural Network with Zero-Allocation Pre-Buffered Tensors.
///
/// Architecture:
/// Input Layer (d_in) -> [Weights W1, Biases b1] -> Hidden Layer (d_hid) [Activation 1]
/// Hidden Layer (d_hid) -> [Weights W2, Biases b2] -> Output Layer (d_out) [Activation 2]

use crate::activation::Activation;
use crate::matrix::{Matrix, SimpleRng};

#[allow(dead_code)]
pub struct TwoLayerNN {

    pub in_dim: usize,
    pub hid_dim: usize,
    pub out_dim: usize,

    // Learnable Parameters (Weights and Balances/Biases)
    pub w1: Matrix,
    pub b1: Matrix,
    pub w2: Matrix,
    pub b2: Matrix,

    // Activation functions
    pub act_hidden: Activation,
    pub act_output: Activation,

    // Pre-allocated forward cache (reused across batches to prevent dynamic heap allocations)
    z1: Matrix,
    pub a1: Matrix,
    z2: Matrix,
    pub a2: Matrix,

    // Pre-allocated backward gradient cache
    delta2: Matrix,
    grad_w2: Matrix,
    grad_b2: Matrix,
    delta1_temp: Matrix,
    act1_deriv: Matrix,
    delta1: Matrix,
    grad_w1: Matrix,
    grad_b1: Matrix,

    // Optimizer Momentum Velocity Buffers
    v_w1: Matrix,
    v_b1: Matrix,
    v_w2: Matrix,
    v_b2: Matrix,
}

impl TwoLayerNN {
    /// Creates a new 2-layer neural network with specified dimensions and activations
    pub fn new(
        in_dim: usize,
        hid_dim: usize,
        out_dim: usize,
        act_hidden: Activation,
        act_output: Activation,
        seed: u64,
    ) -> Self {
        let mut rng = SimpleRng::new(seed);

        // Weight Initialization: He for ReLU, Xavier for Sigmoid/Tanh
        let w1 = match act_hidden {
            Activation::ReLU => Matrix::he_init(in_dim, hid_dim, &mut rng),
            _ => Matrix::xavier_init(in_dim, hid_dim, &mut rng),
        };
        let b1 = Matrix::zeros(1, hid_dim);

        let w2 = match act_output {
            Activation::ReLU => Matrix::he_init(hid_dim, out_dim, &mut rng),
            _ => Matrix::xavier_init(hid_dim, out_dim, &mut rng),
        };
        let b2 = Matrix::zeros(1, out_dim);

        Self {
            in_dim,
            hid_dim,
            out_dim,
            w1,
            b1,
            w2,
            b2,
            act_hidden,
            act_output,

            z1: Matrix::zeros(1, hid_dim),
            a1: Matrix::zeros(1, hid_dim),
            z2: Matrix::zeros(1, out_dim),
            a2: Matrix::zeros(1, out_dim),

            delta2: Matrix::zeros(1, out_dim),
            grad_w2: Matrix::zeros(hid_dim, out_dim),
            grad_b2: Matrix::zeros(1, out_dim),
            delta1_temp: Matrix::zeros(1, hid_dim),
            act1_deriv: Matrix::zeros(1, hid_dim),
            delta1: Matrix::zeros(1, hid_dim),
            grad_w1: Matrix::zeros(in_dim, hid_dim),
            grad_b1: Matrix::zeros(1, hid_dim),

            v_w1: Matrix::zeros(in_dim, hid_dim),
            v_b1: Matrix::zeros(1, hid_dim),
            v_w2: Matrix::zeros(hid_dim, out_dim),
            v_b2: Matrix::zeros(1, out_dim),
        }
    }

    /// Forward pass: Propagates input through Layer 1 and Layer 2
    ///
    /// Mathematical Steps:
    /// 1. Z1 = X * W1 + b1
    /// 2. A1 = sigma(Z1)
    /// 3. Z2 = A1 * W2 + b2
    /// 4. A2 = phi(Z2)
    pub fn forward(&mut self, x: &Matrix) -> &Matrix {
        assert_eq!(x.cols, self.in_dim, "Input feature dimension mismatch");

        // 1. Layer 1 Linear transformation: Z1 = X * W1
        x.matmul(&self.w1, &mut self.z1);
        // Add Layer 1 Bias (Balances): Z1 = Z1 + b1
        self.z1.add_bias_inplace(&self.b1);
        // Layer 1 Non-Linear Activation: A1 = sigma(Z1)
        self.act_hidden.forward(&self.z1, &mut self.a1);

        // 2. Layer 2 Linear transformation: Z2 = A1 * W2
        self.a1.matmul(&self.w2, &mut self.z2);
        // Add Layer 2 Bias (Balances): Z2 = Z2 + b2
        self.z2.add_bias_inplace(&self.b2);
        // Layer 2 Output Activation: A2 = phi(Z2)
        self.act_output.forward(&self.z2, &mut self.a2);

        &self.a2
    }

    /// Backward pass (Backpropagation): Computes analytical gradients with respect to all weights and biases
    ///
    /// Mathematical Steps:
    /// 1. delta2 = (A2 - Y) * phi'(Z2)   [for MSE / identity output]
    /// 2. grad_W2 = A1^T * delta2
    /// 3. grad_b2 = sum_rows(delta2)
    /// 4. delta1 = (delta2 * W2^T) (*) sigma'(Z1)
    /// 5. grad_W1 = X^T * delta1
    /// 6. grad_b1 = sum_rows(delta1)
    pub fn backward(&mut self, x: &Matrix, targets: &Matrix) {
        let batch_size = x.rows as f32;

        // Output error: delta2 = A2 - Y
        self.delta2.reshape_or_realloc(self.a2.rows, self.a2.cols);
        for i in 0..self.a2.data.len() {
            self.delta2.data[i] = self.a2.data[i] - targets.data[i];
        }

        // If output activation is non-linear (e.g. Sigmoid/Tanh with MSE), multiply by phi'(Z2)
        if self.act_output != Activation::Linear {
            self.act_output.derivative(&self.z2, &self.a2, &mut self.delta1_temp);
            for i in 0..self.delta2.data.len() {
                self.delta2.data[i] *= self.delta1_temp.data[i];
            }
        }


        // Layer 2 Gradients: grad_W2 = A1^T * delta2
        self.a1.matmul_transpose_a(&self.delta2, &mut self.grad_w2);
        // grad_b2 = sum across batch rows
        self.delta2.sum_columns_into(&mut self.grad_b2);

        // Hidden Layer Error Propagation: delta1 = (delta2 * W2^T) (*) sigma'(Z1)
        self.delta2.matmul_transpose_b(&self.w2, &mut self.delta1_temp);
        self.act_hidden.derivative(&self.z1, &self.a1, &mut self.act1_deriv);
        self.delta1_temp.hadamard_into(&self.act1_deriv, &mut self.delta1);

        // Layer 1 Gradients: grad_W1 = X^T * delta1
        x.matmul_transpose_a(&self.delta1, &mut self.grad_w1);
        // grad_b1 = sum across batch rows
        self.delta1.sum_columns_into(&mut self.grad_b1);

        // Scale gradients by 1 / batch_size
        let inv_batch = 1.0 / batch_size;
        self.grad_w1.scale_inplace(inv_batch);
        self.grad_b1.scale_inplace(inv_batch);
        self.grad_w2.scale_inplace(inv_batch);
        self.grad_b2.scale_inplace(inv_batch);
    }

    /// Optimizer step: Updates Weights and Biases using SGD with Momentum
    pub fn step_momentum(&mut self, learning_rate: f32, momentum: f32, weight_decay: f32) {
        // Update Layer 1 Weights: v = momentum * v + lr * (grad + wd * w); w -= v
        for i in 0..self.w1.data.len() {
            let g = self.grad_w1.data[i] + weight_decay * self.w1.data[i];
            self.v_w1.data[i] = momentum * self.v_w1.data[i] + learning_rate * g;
            self.w1.data[i] -= self.v_w1.data[i];
        }

        // Update Layer 1 Biases
        for i in 0..self.b1.data.len() {
            let g = self.grad_b1.data[i];
            self.v_b1.data[i] = momentum * self.v_b1.data[i] + learning_rate * g;
            self.b1.data[i] -= self.v_b1.data[i];
        }

        // Update Layer 2 Weights
        for i in 0..self.w2.data.len() {
            let g = self.grad_w2.data[i] + weight_decay * self.w2.data[i];
            self.v_w2.data[i] = momentum * self.v_w2.data[i] + learning_rate * g;
            self.w2.data[i] -= self.v_w2.data[i];
        }

        // Update Layer 2 Biases
        for i in 0..self.b2.data.len() {
            let g = self.grad_b2.data[i];
            self.v_b2.data[i] = momentum * self.v_b2.data[i] + learning_rate * g;
            self.b2.data[i] -= self.v_b2.data[i];
        }
    }

    /// Computes Mean Squared Error (MSE) loss
    pub fn compute_mse(&self, targets: &Matrix) -> f32 {
        let total = self.a2.data.len() as f32;
        let mut sum_sq = 0.0;
        for (pred, targ) in self.a2.data.iter().zip(&targets.data) {
            let diff = pred - targ;
            sum_sq += diff * diff;
        }
        sum_sq / total
    }

    /// High performance train step (Forward -> Backward -> Optimize)
    pub fn train_step(
        &mut self,
        inputs: &Matrix,
        targets: &Matrix,
        lr: f32,
        momentum: f32,
        weight_decay: f32,
    ) -> f32 {
        self.forward(inputs);
        let loss = self.compute_mse(targets);
        self.backward(inputs, targets);
        self.step_momentum(lr, momentum, weight_decay);
        loss
    }
}
