/// High-Performance Matrix Engine optimized for Cache Locality and SIMD Autovectorization.
/// Uses a flat, contiguous 1D buffer in row-major order with zero heap reallocation.

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct Matrix {

    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x853c49e6748fea9b } else { seed },
        }
    }

    /// Generates next pseudo-random u64 using SplitMix64
    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// Generates next f32 in [0, 1)
    #[inline(always)]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }

    /// Box-Muller transform for standard Normal distribution N(0, 1)
    pub fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-7);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

#[allow(dead_code)]
impl Matrix {

    /// Creates a zero-filled matrix of shape (rows, cols)
    #[inline]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Creates a matrix from existing slice/vector
    #[inline]
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(rows * cols, data.len(), "Dimension mismatch in Matrix::from_vec");
        Self { rows, cols, data }
    }

    /// Resizes / re-initializes buffer without re-allocating if capacity permits
    #[inline]
    pub fn reshape_or_realloc(&mut self, rows: usize, cols: usize) {
        self.rows = rows;
        self.cols = cols;
        let total = rows * cols;
        if self.data.len() != total {
            self.data.resize(total, 0.0);
        }
    }

    /// Initializes weights using He / Kaiming Normal initialization (optimized for ReLU)
    pub fn he_init(rows: usize, cols: usize, rng: &mut SimpleRng) -> Self {
        let std_dev = (2.0 / rows as f32).sqrt();
        let data: Vec<f32> = (0..rows * cols).map(|_| rng.next_gaussian() * std_dev).collect();
        Self { rows, cols, data }
    }

    /// Initializes weights using Xavier / Glorot Normal initialization (optimized for Sigmoid/Tanh)
    pub fn xavier_init(rows: usize, cols: usize, rng: &mut SimpleRng) -> Self {
        let std_dev = (2.0 / (rows + cols) as f32).sqrt();
        let data: Vec<f32> = (0..rows * cols).map(|_| rng.next_gaussian() * std_dev).collect();
        Self { rows, cols, data }
    }

    /// High-performance matrix multiplication: out = A (self) * B
    /// Uses cache-aware sequential streaming to maximize L1/L2 cache hits and SIMD pipelining.
    #[inline]
    pub fn matmul(&self, other: &Matrix, out: &mut Matrix) {
        assert_eq!(self.cols, other.rows, "Matrix multiply dimension mismatch ({}x{} * {}x{})", self.rows, self.cols, other.rows, other.cols);
        out.reshape_or_realloc(self.rows, other.cols);
        out.data.fill(0.0);

        let m = self.rows;
        let k = self.cols;
        let n = other.cols;

        let a = &self.data;
        let b = &other.data;
        let c = &mut out.data;

        // Optimized Cache-Friendly GEMM Kernel (i-k-j loop order for continuous row-major memory writes)
        for i in 0..m {
            let a_row_offset = i * k;
            let c_row_offset = i * n;
            for p in 0..k {
                let a_val = a[a_row_offset + p];
                let b_row_offset = p * n;
                
                // Vectorizable inner loop
                for j in 0..n {
                    c[c_row_offset + j] += a_val * b[b_row_offset + j];
                }
            }
        }
    }

    /// High-performance Transpose-Multiply: out = A^T * B
    #[inline]
    pub fn matmul_transpose_a(&self, other: &Matrix, out: &mut Matrix) {
        assert_eq!(self.rows, other.rows, "Transpose A multiply mismatch ({}x{}^T * {}x{})", self.rows, self.cols, other.rows, other.cols);
        out.reshape_or_realloc(self.cols, other.cols);
        out.data.fill(0.0);

        let k = self.rows;
        let m = self.cols;
        let n = other.cols;

        let a = &self.data;
        let b = &other.data;
        let c = &mut out.data;

        for i in 0..m {
            let c_row_offset = i * n;
            for p in 0..k {
                let a_val = a[p * m + i];
                let b_row_offset = p * n;
                for j in 0..n {
                    c[c_row_offset + j] += a_val * b[b_row_offset + j];
                }
            }
        }
    }


    /// High-performance Transpose-Multiply: out = A * B^T
    #[inline]
    pub fn matmul_transpose_b(&self, other: &Matrix, out: &mut Matrix) {
        assert_eq!(self.cols, other.cols, "Transpose B multiply mismatch ({}x{} * {}x{}^T)", self.rows, self.cols, other.rows, other.cols);
        out.reshape_or_realloc(self.rows, other.rows);

        let m = self.rows;
        let k = self.cols;
        let n = other.rows;

        let a = &self.data;
        let b = &other.data;
        let c = &mut out.data;

        for i in 0..m {
            let a_row_offset = i * k;
            let c_row_offset = i * n;
            for j in 0..n {
                let b_row_offset = j * k;
                let mut sum = 0.0;
                // Dot product of row i of A and row j of B (both contiguous in memory!)
                for p in 0..k {
                    sum += a[a_row_offset + p] * b[b_row_offset + p];
                }
                c[c_row_offset + j] = sum;
            }
        }
    }

    /// Adds a 1-row bias vector to every row of the matrix in-place
    #[inline]
    pub fn add_bias_inplace(&mut self, bias: &Matrix) {
        assert_eq!(bias.rows, 1, "Bias must have 1 row");
        assert_eq!(self.cols, bias.cols, "Bias cols must match matrix cols");

        let b = &bias.data;
        let cols = self.cols;
        for (i, val) in self.data.iter_mut().enumerate() {
            *val += b[i % cols];
        }
    }

    /// Accumulates column sums of this matrix into a (1, cols) bias gradient matrix
    #[inline]
    pub fn sum_columns_into(&self, bias_grad: &mut Matrix) {
        bias_grad.reshape_or_realloc(1, self.cols);
        bias_grad.data.fill(0.0);

        let cols = self.cols;
        let bg = &mut bias_grad.data;
        for (i, val) in self.data.iter().enumerate() {
            bg[i % cols] += *val;
        }
    }

    /// Element-wise Hadamard Product: out[i] = self[i] * other[i]
    #[inline]
    pub fn hadamard_into(&self, other: &Matrix, out: &mut Matrix) {
        assert_eq!(self.data.len(), other.data.len(), "Hadamard length mismatch");
        out.reshape_or_realloc(self.rows, self.cols);

        let d1 = &self.data;
        let d2 = &other.data;
        let o = &mut out.data;

        for i in 0..d1.len() {
            o[i] = d1[i] * d2[i];
        }
    }

    /// In-place element-wise subtraction: self = self - other
    #[inline]
    pub fn sub_inplace(&mut self, other: &Matrix) {
        assert_eq!(self.data.len(), other.data.len());
        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a -= *b;
        }
    }

    /// In-place scaling: self = self * scalar
    #[inline]
    pub fn scale_inplace(&mut self, scalar: f32) {
        for val in self.data.iter_mut() {
            *val *= scalar;
        }
    }

    /// In-place addition with scaling: self += other * scalar
    #[inline]
    pub fn add_scaled_inplace(&mut self, other: &Matrix, scalar: f32) {
        assert_eq!(self.data.len(), other.data.len());
        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a += *b * scalar;
        }
    }
}
