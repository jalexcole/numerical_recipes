use std::f64;

fn sign(x: f64, y: f64) -> f64 {
    // Returns |x| with the sign of y.
    if y >= 0.0 { x.abs() } else { -x.abs() }
}

/// Object for QR decomposition of a square matrix A and related functions.
#[derive(Debug, Clone)]
pub struct QRdcmp {
    n: usize,
    /// Stored Qᵀ matrix (n×n).
    qt: Vec<Vec<f64>>,
    /// Stored R matrix (n×n, upper-triangular).
    r: Vec<Vec<f64>>,
    /// True if a singularity was detected during decomposition.
    sing: bool,
}

impl QRdcmp {
    /// Constructs the QR decomposition of a square matrix `a` (n×n).
    ///
    /// The decomposition is performed so that A = Q R, but R and Qᵀ are stored.
    /// If a singularity is encountered during the process, `sing` is set to true.
    ///
    /// # Panics
    /// Panics if `a` is not square.
    pub fn new(a: &Vec<Vec<f64>>) -> Self {
        let n = a.len();
        if n == 0 || a.iter().any(|row| row.len() != n) {
            panic!("QRdcmp: need square matrix");
        }
        let mut r = a.clone();
        // Initialize qt as an n×n identity matrix.
        let mut qt = vec![vec![0.0; n]; n];
        for i in 0..n {
            qt[i][i] = 1.0;
        }
        let mut sing = false;
        let mut c = vec![0.0; n];
        let mut d = vec![0.0; n];

        // Main loop over columns k = 0 .. n-2.
        for k in 0..(n - 1) {
            let mut scale = 0.0;
            for i in k..n {
                scale = f64::max(scale, r[i][k].abs());
            }
            if scale == 0.0 {
                sing = true;
                c[k] = 0.0;
                d[k] = 0.0;
            } else {
                for i in k..n {
                    r[i][k] /= scale;
                }
                let mut sum = 0.0;
                for i in k..n {
                    sum += r[i][k] * r[i][k];
                }
                let sigma = sign(sum.sqrt(), r[k][k]);
                r[k][k] += sigma;
                c[k] = sigma * r[k][k];
                d[k] = -scale * sigma;
                for j in (k + 1)..n {
                    let mut sum = 0.0;
                    for i in k..n {
                        sum += r[i][k] * r[i][j];
                    }
                    let tau = sum / c[k];
                    for i in k..n {
                        r[i][j] -= tau * r[i][k];
                    }
                }
            }
        }
        d[n - 1] = r[n - 1][n - 1];
        if d[n - 1] == 0.0 {
            sing = true;
        }
        // Form QT explicitly.
        // (qt was initialized as identity above.)
        for k in 0..(n - 1) {
            if c[k] != 0.0 {
                for j in 0..n {
                    let mut sum = 0.0;
                    for i in k..n {
                        sum += r[i][k] * qt[i][j];
                    }
                    sum /= c[k];
                    for i in k..n {
                        qt[i][j] -= sum * r[i][k];
                    }
                }
            }
        }
        // Form R explicitly: set diagonal from d and zero out lower-triangular part.
        for i in 0..n {
            r[i][i] = d[i];
            for j in 0..i {
                r[i][j] = 0.0;
            }
        }
        QRdcmp { n, qt, r, sing }
    }

    /// Solves the linear system A·x = b using the stored QR decomposition.
    ///
    /// The solution is computed by forming Qᵀ·b and then back-solving R·x = Qᵀ·b.
    ///
    /// # Panics
    /// Panics if the size of b or x is not n.
    pub fn solve(&self, b: &[f64], x: &mut [f64]) {
        if b.len() != self.n || x.len() != self.n {
            panic!("QRdcmp::solve: bad sizes");
        }
        // First, compute x = Qᵀ·b.
        self.qtmult(b, x);
        // Now, solve R·x = (Qᵀ·b) for x.
        let tmp = x.to_vec();
        self.rsolve(&tmp, x);
    }

    /// Multiplies Qᵀ by vector b and stores the result in x.
    /// Since Q is orthogonal, Qᵀ·b is equivalent to solving Q·x = b.
    pub fn qtmult(&self, b: &[f64], x: &mut [f64]) {
        if b.len() != self.n || x.len() != self.n {
            panic!("QRdcmp::qtmult: bad sizes");
        }
        for i in 0..self.n {
            let mut sum = 0.0;
            for j in 0..self.n {
                sum += self.qt[i][j] * b[j];
            }
            x[i] = sum;
        }
    }

    /// Solves the triangular system R·x = b by back-substitution.
    /// The solution is stored in x.
    ///
    /// # Panics
    /// Panics if the matrix was detected as singular.
    pub fn rsolve(&self, b: &[f64], x: &mut [f64]) {
        if self.sing {
            panic!("QRdcmp::rsolve: attempting solve in a singular QR");
        }
        if b.len() != self.n || x.len() != self.n {
            panic!("QRdcmp::rsolve: bad sizes");
        }
        for i in (0..self.n).rev() {
            let mut sum = b[i];
            for j in (i + 1)..self.n {
                sum -= self.r[i][j] * x[j];
            }
            x[i] = sum / self.r[i][i];
        }
    }

    /// Updates the stored QR decomposition from that of A to the QR decomposition
    /// of the matrix Q · (R + u·vᵀ).
    ///
    /// Here `u` and `v` are input vectors (each of length n). This routine updates the
    /// stored matrices in place.
    pub fn update(&mut self, u: &[f64], v: &[f64]) {
        if u.len() != self.n || v.len() != self.n {
            panic!("QRdcmp::update: bad sizes for u or v");
        }
        // Make a mutable copy of u into vector w.
        let mut w = u.to_vec();
        // Find the largest index k such that w[k] is nonzero, scanning backward.
        let mut k_opt = None;
        for idx in (0..self.n).rev() {
            if w[idx] != 0.0 {
                k_opt = Some(idx);
                break;
            }
        }
        let k = k_opt.unwrap_or(0);

        // For i from k-1 downto 0, update w and apply rotations.
        if k > 0 {
            for i in (0..k).rev() {
                self.rotate(i, w[i], -w[i + 1]);
                if w[i] == 0.0 {
                    w[i] = w[i + 1].abs();
                } else if w[i].abs() > w[i + 1].abs() {
                    w[i] = w[i].abs() * (1.0 + (w[i + 1] / w[i]).powi(2)).sqrt();
                } else {
                    w[i] = w[i + 1].abs() * (1.0 + (w[i] / w[i + 1]).powi(2)).sqrt();
                }
            }
        }
        // Update first row of R.
        for i in 0..self.n {
            self.r[0][i] += w[0] * v[i];
        }
        // For i = 0..k-1, perform rotations on R.
        for i in 0..k {
            self.rotate(i, self.r[i][i], -self.r[i + 1][i]);
        }
        // Check for zeros on the diagonal.
        for i in 0..self.n {
            if self.r[i][i] == 0.0 {
                self.sing = true;
            }
        }
    }

    /// Performs a Jacobi rotation on rows i and i+1 of both R and QT.
    ///
    /// The rotation parameters are computed from a and b:
    ///     cos = 1/sqrt(1 + (fact)²) with sign of a, and sin = fact*cos,
    /// where fact = b/a or a/b, chosen to avoid overflow.
    pub fn rotate(&mut self, i: usize, a: f64, b: f64) {
        let mut c: f64;
        let mut s: f64;
        if a == 0.0 {
            c = 0.0;
            s = if b >= 0.0 { 1.0 } else { -1.0 };
        } else if a.abs() > b.abs() {
            let fact = b / a;
            let temp = (1.0 + fact * fact).sqrt();
            c = sign(1.0 / temp, a);
            s = fact * c;
        } else {
            let fact = a / b;
            let temp = (1.0 + fact * fact).sqrt();
            s = sign(1.0 / temp, b);
            c = fact * s;
        }
        // Rotate rows i and i+1 of R.
        for j in i..self.n {
            let y = self.r[i][j];
            let w = self.r[i + 1][j];
            self.r[i][j] = c * y - s * w;
            self.r[i + 1][j] = s * y + c * w;
        }
        // Rotate rows i and i+1 of QT.
        for j in 0..self.n {
            let y = self.qt[i][j];
            let w = self.qt[i + 1][j];
            self.qt[i][j] = c * y - s * w;
            self.qt[i + 1][j] = s * y + c * w;
        }
    }
}

fn main() {
    // --- Example usage of QRdcmp ---
    //
    // Consider a 3×3 matrix A.
    let a = vec![
        vec![12.0, -51.0, 4.0],
        vec![6.0, 167.0, -68.0],
        vec![-4.0, 24.0, -41.0],
    ];
    let qr = QRdcmp::new(&a);
    println!("QR decomposition:");
    println!("Matrix R:");
    for row in qr.r.iter() {
        println!("{:?}", row);
    }
    println!("Matrix Qᵀ:");
    for row in qr.qt.iter() {
        println!("{:?}", row);
    }
    println!("Singular flag: {}", qr.sing);

    // Solve A·x = b.
    let b = vec![1.0, 2.0, 3.0];
    let mut x = vec![0.0; 3];
    // Note: We use the QRdcmp object to solve; the solution is computed via Qᵀ and R.
    let mut qr2 = qr.clone(); // Use a clone since solve() doesn't change the decomposition.
    qr2.solve(&b, &mut x);
    println!("Solution x for A·x = b: {:?}", x);

    // (For a complete test, one might also call update and rotate,
    // but these are more advanced operations.)
}
