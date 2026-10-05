/// Activation Functions and their analytical derivatives for Fast Forward & Backward Passes.

use crate::matrix::Matrix;

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Activation {

    ReLU,
    Sigmoid,
    Tanh,
    Linear,
}

impl Activation {
    /// Applies the activation function element-wise: out = f(src)
    pub fn forward(&self, src: &Matrix, out: &mut Matrix) {
        out.reshape_or_realloc(src.rows, src.cols);
        let s = &src.data;
        let d = &mut out.data;

        match self {
            Activation::ReLU => {
                for i in 0..s.len() {
                    d[i] = if s[i] > 0.0 { s[i] } else { 0.0 };
                }
            }
            Activation::Sigmoid => {
                for i in 0..s.len() {
                    // Stable sigmoid
                    let x = s[i];
                    if x >= 0.0 {
                        let exp_neg = (-x).exp();
                        d[i] = 1.0 / (1.0 + exp_neg);
                    } else {
                        let exp_pos = x.exp();
                        d[i] = exp_pos / (1.0 + exp_pos);
                    }
                }
            }
            Activation::Tanh => {
                for i in 0..s.len() {
                    d[i] = s[i].tanh();
                }
            }
            Activation::Linear => {
                d.copy_from_slice(s);
            }
        }
    }

    /// Computes the derivative element-wise with respect to pre-activation Z or post-activation A
    /// `out[i] = f'(z[i])`
    pub fn derivative(&self, pre_act_z: &Matrix, post_act_a: &Matrix, out: &mut Matrix) {
        out.reshape_or_realloc(pre_act_z.rows, pre_act_z.cols);
        let z = &pre_act_z.data;
        let a = &post_act_a.data;
        let d = &mut out.data;

        match self {
            Activation::ReLU => {
                for i in 0..z.len() {
                    d[i] = if z[i] > 0.0 { 1.0 } else { 0.0 };
                }
            }
            Activation::Sigmoid => {
                for i in 0..a.len() {
                    let sig = a[i];
                    d[i] = sig * (1.0 - sig);
                }
            }
            Activation::Tanh => {
                for i in 0..a.len() {
                    let th = a[i];
                    d[i] = 1.0 - th * th;
                }
            }
            Activation::Linear => {
                d.fill(1.0);
            }
        }
    }
}
