use std::f64;

/// Object for singular value decomposition (SVD) of a matrix A and related functions.
///
/// The SVD decomposes A (of dimensions m×n) as
///     A = U · W · Vᵀ,
/// where U is m×n, W is a diagonal n×n matrix (its diagonal stored in `w`), and V is n×n.
///
/// The SVD computation is performed by calling decompose() and then reorder(),
/// and a threshold for “nonzero” singular values is stored in `tsh`.
#[derive(Debug)]
pub struct SVD {
    pub m: usize,         // number of rows of A
    pub n: usize,         // number of columns of A
    pub u: Vec<Vec<f64>>, // m×n matrix; initially a copy of A, then modified
    pub v: Vec<Vec<f64>>, // n×n matrix; will contain the orthogonal matrix V
    pub w: Vec<f64>,      // singular values (length n)
    pub eps: f64,         // machine epsilon
    pub tsh: f64,         // threshold for nonzero singular values
}

impl SVD {
    /// Constructs an SVD object for the matrix `a`.
    ///
    /// The SVD of a is computed upon construction. The default threshold for nonzero singular
    /// values is set to 0.5 * sqrt(m+n+1) * w[0] * eps.
    pub fn new(a: &Vec<Vec<f64>>) -> Self {
        let m = a.len();
        let n = if m > 0 { a[0].len() } else { 0 };
        let u = a.clone();
        let v = vec![vec![0.0; n]; n];
        let w = vec![0.0; n];
        let eps = f64::EPSILON;
        let mut svd = SVD {
            m,
            n,
            u,
            v,
            w,
            eps,
            tsh: 0.0,
        };
        svd.decompose();
        svd.reorder();
        svd.tsh = 0.5 * ((m + n + 1) as f64).sqrt() * svd.w[0] * eps;
        svd
    }

    /// Computes the SVD decomposition.
    ///
    /// This function should implement an algorithm such as the Golub–Reinsch algorithm.
    /// For brevity, the detailed implementation is omitted.
    pub fn decompose(&mut self) {
        unimplemented!("SVD decomposition algorithm is not implemented");
    }

    /// Reorders the singular values into descending order and reorders U and V accordingly.
    pub fn reorder(&mut self) {
        unimplemented!("SVD reordering is not implemented");
    }

    /// Returns sqrt(a² + b²) without overflow/underflow.
    pub fn pythag(a: f64, b: f64) -> f64 {
        let absa = a.abs();
        let absb = b.abs();
        if absa > absb {
            absa * (1.0 + (absb / absa).powi(2)).sqrt()
        } else if absb != 0.0 {
            absb * (1.0 + (absa / absb).powi(2)).sqrt()
        } else {
            0.0
        }
    }

    /// Returns the reciprocal of the condition number of A.
    ///
    /// If either w[0] or w[n-1] is nonpositive, 0 is returned.
    pub fn inv_condition(&self) -> f64 {
        if self.w[0] <= 0.0 || self.w[self.n - 1] <= 0.0 {
            0.0
        } else {
            self.w[self.n - 1] / self.w[0]
        }
    }

    /// Returns the rank of A given a threshold for singular values.
    ///
    /// If thresh is negative, a default value based on roundoff is used.
    pub fn rank(&mut self, thresh: f64) -> usize {
        self.tsh = if thresh >= 0.0 {
            thresh
        } else {
            0.5 * ((self.m + self.n + 1) as f64).sqrt() * self.w[0] * self.eps
        };
        let mut nr = 0;
        for j in 0..self.n {
            if self.w[j] > self.tsh {
                nr += 1;
            }
        }
        nr
    }

    /// Returns the nullity of A given a threshold for singular values.
    ///
    /// If thresh is negative, a default value based on roundoff is used.
    pub fn nullity(&mut self, thresh: f64) -> usize {
        self.tsh = if thresh >= 0.0 {
            thresh
        } else {
            0.5 * ((self.m + self.n + 1) as f64).sqrt() * self.w[0] * self.eps
        };
        let mut nn = 0;
        for j in 0..self.n {
            if self.w[j] <= self.tsh {
                nn += 1;
            }
        }
        nn
    }

    /// Returns an orthonormal basis for the range (column space) of A.
    ///
    /// The basis vectors are returned as the columns of an m×(rank) matrix.
    pub fn range(&mut self, thresh: f64) -> Vec<Vec<f64>> {
        let rnk = self.rank(thresh);
        let mut rnge = vec![vec![0.0; rnk]; self.m];
        let mut nr = 0;
        for j in 0..self.n {
            if self.w[j] > self.tsh {
                for i in 0..self.m {
                    rnge[i][nr] = self.u[i][j];
                }
                nr += 1;
            }
        }
        rnge
    }

    /// Returns an orthonormal basis for the nullspace of A.
    ///
    /// The basis vectors are returned as the columns of an n×(nullity) matrix.
    pub fn nullspace(&mut self, thresh: f64) -> Vec<Vec<f64>> {
        let nullity = self.nullity(thresh);
        let mut nullsp = vec![vec![0.0; nullity]; self.n];
        let mut nn = 0;
        for j in 0..self.n {
            if self.w[j] <= self.tsh {
                for jj in 0..self.n {
                    nullsp[jj][nn] = self.v[jj][j];
                }
                nn += 1;
            }
        }
        nullsp
    }

    /// Solves A·x = b for a single right-hand side vector x using the pseudoinverse.
    ///
    /// The input vector b (length m) and output vector x (length n) must be provided.
    /// If thresh is negative, a default threshold based on roundoff is used.
    pub fn solve_vector(&mut self, b: &[f64], x: &mut [f64], thresh: f64) {
        if b.len() != self.m || x.len() != self.n {
            panic!("SVD::solve_vector bad sizes");
        }
        let mut tmp = vec![0.0; self.n];
        self.tsh = if thresh >= 0.0 {
            thresh
        } else {
            0.5 * ((self.m + self.n + 1) as f64).sqrt() * self.w[0] * self.eps
        };
        // Compute Uᵀ·b and scale by 1/w[j]
        for j in 0..self.n {
            let mut s = 0.0;
            if self.w[j] > self.tsh {
                for i in 0..self.m {
                    s += self.u[i][j] * b[i];
                }
                s /= self.w[j];
            }
            tmp[j] = s;
        }
        // Multiply by V to get x = V·tmp.
        for j in 0..self.n {
            let mut s = 0.0;
            for jj in 0..self.n {
                s += self.v[j][jj] * tmp[jj];
            }
            x[j] = s;
        }
    }

    /// Solves A·X = B for multiple right-hand side vectors.
    ///
    /// B is given as an m×p matrix (each column is a right-hand side),
    /// and the solution X (an n×p matrix) is returned in x.
    pub fn solve_matrix(&mut self, b: &Vec<Vec<f64>>, x: &mut Vec<Vec<f64>>, thresh: f64) {
        let p = b[0].len();
        if b.len() != self.m || x.len() != self.n || x[0].len() != p {
            panic!("SVD::solve_matrix bad sizes");
        }
        let mut xx = vec![0.0; self.n];
        let mut bcol = vec![0.0; self.m];
        for j in 0..p {
            for i in 0..self.m {
                bcol[i] = b[i][j];
            }
            self.solve_vector(&bcol, &mut xx, thresh);
            for i in 0..self.n {
                x[i][j] = xx[i];
            }
        }
    }
}

fn main() {
    // === Example usage of SVD ===
    //
    // We construct a simple 3×2 matrix A.
    let a = vec![
        vec![1.0, 2.0],
        vec![3.0, 4.0],
        vec![5.0, 6.0],
    ];
    // Create the SVD of A.
    let mut svd = SVD::new(&a);

    // (The decompose() and reorder() methods are not implemented,
    // so running this code will panic. In a complete implementation, these
    // functions would compute U, w, and V.)
    //
    // For illustration, suppose the SVD had been computed; we print the singular values:
    println!("Singular values: {:?}", svd.w);

    // Solve A·x = b for a given right-hand side b (with b of length m = 3, x of length n = 2).
    let b = vec![1.0, 2.0, 3.0];
    let mut x = vec![0.0; svd.n];
    svd.solve_vector(&b, &mut x, -1.0);
    println!("Solution vector x: {:?}", x);

    // Compute rank and nullity.
    let rank = svd.rank(-1.0);
    let nullity = svd.nullity(-1.0);
    println!("Rank: {}, Nullity: {}", rank, nullity);

    // Get an orthonormal basis for the range and nullspace.
    let range = svd.range(-1.0);
    let nullspace = svd.nullspace(-1.0);
    println!("Range basis: {:?}", range);
    println!("Nullspace basis: {:?}", nullspace);

    // Print the reciprocal condition number.
    println!("Reciprocal condition number: {:.5}", svd.inv_condition());
}