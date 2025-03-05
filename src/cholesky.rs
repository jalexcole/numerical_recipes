use std::f64;

#[derive(Debug, Clone)]
pub struct Cholesky {
    n: usize,
    /// The lower-triangular factor L of the decomposition.
    /// After decomposition, A = L·Lᵀ.
    /// Only the lower-triangular part is used; the upper-triangular entries are zero.
    el: Vec<Vec<f64>>,
}

impl Cholesky {
    /// Constructs the Cholesky decomposition of a positive-definite symmetric matrix `a`.
    ///
    /// # Panics
    /// Panics if the input matrix is not square or if the matrix is not positive-definite.
    pub fn new(a: &Vec<Vec<f64>>) -> Self {
        let n = a.len();
        if n == 0 {
            panic!("Matrix is empty");
        }
        // Ensure the matrix is square.
        for row in a.iter() {
            if row.len() != n {
                panic!("need square matrix");
            }
        }
        // Create a working copy of a.
        let mut el = a.clone();

        // Perform Cholesky decomposition.
        for i in 0..n {
            for j in i..n {
                // Compute sum = a[i][j] - sum_{k=0}^{i-1} L[i][k]*L[j][k]
                let mut sum = el[i][j];
                for k in (0..i).rev() {
                    sum -= el[i][k] * el[j][k];
                }
                if i == j {
                    if sum <= 0.0 {
                        panic!("Cholesky failed: matrix not positive definite");
                    }
                    el[i][i] = sum.sqrt();
                } else {
                    el[j][i] = sum / el[i][i];
                }
            }
        }
        // Zero out the upper triangular part.
        for i in 0..n {
            for j in 0..i {
                el[j][i] = 0.0;
            }
        }
        Cholesky { n, el }
    }

    /// Solves the set of linear equations A·x = b,
    /// where A is the original matrix and its Cholesky factorization is stored in `el`.
    /// The vector b is given as input and the solution is returned in x.
    ///
    /// # Panics
    /// Panics if the lengths of `b` or `x` are not equal to n.
    pub fn solve(&self, b: &[f64], x: &mut [f64]) {
        if b.len() != self.n || x.len() != self.n {
            panic!("bad lengths in Cholesky::solve");
        }
        // Forward substitution: solve L·y = b, store y in x.
        for i in 0..self.n {
            let mut sum = b[i];
            for k in 0..i {
                sum -= self.el[i][k] * x[k];
            }
            x[i] = sum / self.el[i][i];
        }
        // Backward substitution: solve Lᵀ·x = y.
        for i in (0..self.n).rev() {
            let mut sum = x[i];
            for k in (i + 1)..self.n {
                sum -= self.el[k][i] * x[k];
            }
            x[i] = sum / self.el[i][i];
        }
    }

    /// Computes the product b = L·y, where L is the lower-triangular factor.
    ///
    /// # Panics
    /// Panics if the lengths of `y` or `b` are not equal to n.
    pub fn elmult(&self, y: &[f64], b: &mut [f64]) {
        if y.len() != self.n || b.len() != self.n {
            panic!("bad lengths in Cholesky::elmult");
        }
        for i in 0..self.n {
            let mut sum = 0.0;
            for j in 0..=i {
                sum += self.el[i][j] * y[j];
            }
            b[i] = sum;
        }
    }

    /// Solves the triangular system L·y = b, where L is the lower-triangular factor.
    /// The right-hand side vector b is given as input and the solution is returned in y.
    ///
    /// # Panics
    /// Panics if the lengths of `b` or `y` are not equal to n.
    pub fn elsolve(&self, b: &[f64], y: &mut [f64]) {
        if b.len() != self.n || y.len() != self.n {
            panic!("bad lengths in Cholesky::elsolve");
        }
        for i in 0..self.n {
            let mut sum = b[i];
            for j in 0..i {
                sum -= self.el[i][j] * y[j];
            }
            y[i] = sum / self.el[i][i];
        }
    }

    /// Computes the inverse of the matrix A (whose Cholesky decomposition is stored).
    /// Returns a new n×n matrix representing A⁻¹.
    pub fn inverse(&self) -> Vec<Vec<f64>> {
        let n = self.n;
        // Allocate ainv as an n x n zero matrix.
        let mut ainv = vec![vec![0.0; n]; n];

        // First loop: compute elements for the upper triangle of the inverse.
        for i in 0..n {
            for j in 0..=i {
                let mut sum = if i == j { 1.0 } else { 0.0 };
                for k in (j..i).rev() {
                    sum -= self.el[i][k] * ainv[j][k];
                }
                ainv[j][i] = sum / self.el[i][i];
            }
        }
        // Second loop: complete the inverse by symmetry.
        for i in (0..n).rev() {
            for j in 0..=i {
                let mut sum = ainv[j][i];
                for k in (i + 1)..n {
                    sum -= self.el[k][i] * ainv[j][k];
                }
                let value = sum / self.el[i][i];
                ainv[i][j] = value;
                ainv[j][i] = value; // The inverse is symmetric.
            }
        }
        ainv
    }

    /// Returns the logarithm of the determinant of A.
    /// (For a Cholesky decomposition, log(det(A)) = 2·∑ log(diag(L)).)
    pub fn logdet(&self) -> f64 {
        let mut sum = 0.0;
        for i in 0..self.n {
            sum += self.el[i][i].ln();
        }
        2.0 * sum
    }
}

fn main() {
    // Example: Compute the Cholesky decomposition of a 3×3 positive-definite symmetric matrix.
    let a = vec![
        vec![4.0, 12.0, -16.0],
        vec![12.0, 37.0, -43.0],
        vec![-16.0, -43.0, 98.0],
    ];
    let chol = Cholesky::new(&a);
    println!("Cholesky decomposition (L):");
    for row in chol.el.iter() {
        println!("{:?}", row);
    }

    // Solve the system A·x = b.
    let b = vec![1.0, 2.0, 3.0];
    let mut x = vec![0.0; 3];
    chol.solve(&b, &mut x);
    println!("Solution x: {:?}", x);

    // Compute L*y for a given y.
    let y = vec![1.0, 1.0, 1.0];
    let mut by = vec![0.0; 3];
    chol.elmult(&y, &mut by);
    println!("L * y: {:?}", by);

    // Solve L*y = b for y.
    let mut ysol = vec![0.0; 3];
    chol.elsolve(&b, &mut ysol);
    println!("Solution y from L*y = b: {:?}", ysol);

    // Compute the inverse of A.
    let ainv = chol.inverse();
    println!("Inverse of A:");
    for row in ainv.iter() {
        println!("{:?}", row);
    }

    // Compute the logarithm of the determinant.
    let log_det = chol.logdet();
    println!("Log determinant of A: {}", log_det);
}