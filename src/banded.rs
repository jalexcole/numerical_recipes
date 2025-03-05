

use std::cmp::{max, min};

/// Multiplies a band‐diagonal matrix A by a vector x, storing the result in b.
/// 
/// The matrix `a` is stored in compact form as an n×(m₁+m₂+1) array:
/// - The diagonal elements are in a[i][m₁].
/// - The subdiagonals are in a[j..n][0..m₁-1] (with a[0] unused).
/// - The superdiagonals are in a[0..j][m₁+1..m₁+m₂].
/// 
/// # Arguments
/// * `a` – Banded matrix with n rows and m₁+m₂+1 columns.
/// * `m1` – Number of subdiagonals.
/// * `m2` – Number of superdiagonals.
/// * `x` – Input vector (length n).
/// * `b` – Output vector (length n) where the product A·x is stored.
pub fn banmul(a: &Vec<Vec<f64>>, m1: usize, m2: usize, x: &[f64], b: &mut [f64]) {
    let n = a.len();
    for i in 0..n {
        // k = i - m1 (can be negative)
        let k = i as isize - m1 as isize;
        // The number of elements in row i available for the multiplication is the lesser of:
        //   (m1+m2+1) and (n - k)   [where n-k is computed in isize arithmetic]
        let tmploop = min(m1 + m2 + 1, (n as isize - k) as usize);
        b[i] = 0.0;
        // j runs from max(0, -k) up to tmploop-1
        let start_j = max(0, -k) as usize;
        for j in start_j..tmploop {
            // In the compact storage, the element a[i][j] multiplies x at index (j + k)
            let x_index = (j as isize + k) as usize;
            b[i] += a[i][j] * x[x_index];
        }
    }
}

/// An object for LU-decomposition of a band-diagonal matrix and solving linear equations.
/// 
/// The band-diagonal matrix A has n rows, with m₁ subdiagonals and m₂ superdiagonals.
/// A is stored compactly in an n×(m₁+m₂+1) array, with the main diagonal in column m₁.
/// 
/// # Fields
/// * `n`    – Number of rows (and columns) of A.
/// * `m1`   – Number of subdiagonals.
/// * `m2`   – Number of superdiagonals.
/// * `au`   – Upper-triangular matrix from the LU decomposition (stored compactly).
/// * `al`   – Lower-triangular multipliers (stored compactly as an n×m₁ array).
/// * `indx` – Permutation vector (stored as 1-indexed pivot indices).
/// * `d`    – +1 or -1 depending on whether the number of row interchanges is even or odd.
#[derive(Debug)]
pub struct Bandec {
    n: usize,
    m1: usize,
    m2: usize,
    au: Vec<Vec<f64>>, // Dimensions: n × (m1+m2+1)
    al: Vec<Vec<f64>>, // Dimensions: n × m1 (only rows that need it)
    indx: Vec<usize>,  // Pivot indices stored as 1-indexed values.
    d: f64,
}

impl Bandec {
    /// Constructs the LU-decomposition of a band-diagonal matrix.
    ///
    /// The input matrix `a` (with dimensions n×(m₁+m₂+1)) is not modified;
    /// its copy is used to compute the LU decomposition. See the comment for `banmul`
    /// for details on the storage format.
    ///
    /// # Arguments
    /// * `a`   – The input band-diagonal matrix.
    /// * `mm1` – The number of subdiagonals.
    /// * `mm2` – The number of superdiagonals.
    pub fn new(a: &Vec<Vec<f64>>, mm1: usize, mm2: usize) -> Self {
        let n = a.len();
        let m1 = mm1;
        let m2 = mm2;
        let mm = m1 + m2 + 1;
        // Make a working copy of a.
        let mut au = a.clone();
        // Allocate storage for the lower triangular part (n rows, each of length m1).
        let mut al = vec![vec![0.0; m1]; n];
        let mut indx = vec![0; n];
        let mut d = 1.0;
        const TINY: f64 = 1.0e-40;

        // Rearrangement of the storage for the first m1 rows.
        let mut l = m1;
        for i in 0..m1 {
            // For j from (m1 - i) to mm-1, move elements left.
            for j in (m1 - i)..mm {
                let target = j - l;
                au[i][target] = au[i][j];
            }
            // Decrement l.
            l = l.saturating_sub(1);
            // Zero out the vacated positions.
            for j in (mm - l - 1)..mm {
                au[i][j] = 0.0;
            }
        }

        d = 1.0;
        l = m1;
        for k in 0..n {
            let mut dum = au[k][0];
            let mut i_max = k;
            if l < n {
                l += 1;
            }
            for j in (k + 1)..l {
                if au[j][0].abs() > dum.abs() {
                    dum = au[j][0];
                    i_max = j;
                }
            }
            // Store pivot index (using 1-indexing to mimic the original code).
            indx[k] = i_max + 1;
            if dum == 0.0 {
                au[k][0] = TINY;
            }
            if i_max != k {
                // Swap entire rows k and i_max.
                au.swap(k, i_max);
                d = -d;
            }
            // Perform elimination for rows k+1 to l-1.
            for i in (k + 1)..l {
                let factor = au[i][0] / au[k][0];
                al[k][i - k - 1] = factor;
                for j in 1..mm {
                    au[i][j - 1] = au[i][j] - factor * au[k][j];
                }
                au[i][mm - 1] = 0.0;
            }
        }

        Bandec { n, m1, m2, au, al, indx, d }
    }

    /// Solves the system A·x = b using the stored LU decomposition.
    ///
    /// The input vector `b` (of length n) is permuted and then solved.
    /// The solution is returned in the vector `x` (which must have length n).
    pub fn solve(&self, b: &[f64], x: &mut [f64]) {
        let mm = self.m1 + self.m2 + 1;
        let n = self.n;
        let mut l = self.m1;

        // Copy b into x.
        for i in 0..n {
            x[i] = b[i];
        }

        // Forward substitution.
        for k in 0..n {
            // Pivot index (convert 1-indexed to 0-indexed).
            let j = self.indx[k] - 1;
            if j != k {
                x.swap(k, j);
            }
            if l < n {
                l += 1;
            }
            for j in (k + 1)..l {
                x[j] -= self.al[k][j - k - 1] * x[k];
            }
        }

        // Backsubstitution.
        l = 1;
        for i in (0..n).rev() {
            let mut dum = x[i];
            for k in 1..l {
                // x[i+k] is within bounds since l ≤ mm and i + k < n by construction.
                dum -= self.au[i][k] * x[i + k];
            }
            x[i] = dum / self.au[i][0];
            if l < mm {
                l += 1;
            }
        }
    }

    /// Returns the determinant of A using the stored LU decomposition.
    pub fn det(&self) -> f64 {
        let mut dd = self.d;
        for i in 0..self.n {
            dd *= self.au[i][0];
        }
        dd
    }
}

fn main() {
    // --- Example usage for banmul ---
    //
    // Suppose we have a 4×4 banded matrix A with m₁ = 1 and m₂ = 1 stored compactly.
    // Its storage has 1+1+1 = 3 columns per row.
    // For example, let
    //   a[0] = [   X,  4.0,  1.0 ]   (X denotes unused; the diagonal element is 4.0)
    //   a[1] = [ 2.0,  5.0,  3.0 ]
    //   a[2] = [ 1.0,  6.0,  4.0 ]
    //   a[3] = [ 0.0,  7.0,    X ]
    // Here the main diagonal is in column index 1.
    let a_band: Vec<Vec<f64>> = vec![
        vec![0.0, 4.0, 1.0],
        vec![2.0, 5.0, 3.0],
        vec![1.0, 6.0, 4.0],
        vec![0.0, 7.0, 0.0],
    ];
    let x_vec = vec![1.0, 2.0, 3.0, 4.0];
    let mut b_vec = vec![0.0; 4];
    banmul(&a_band, 1, 1, &x_vec, &mut b_vec);
    println!("Result of banmul, b = {:?}", b_vec);

    // --- Example usage for Bandec ---
    //
    // We now create a banded matrix and decompose it.
    // For demonstration, we use a 3×3 matrix stored in compact form with m₁ = 1 and m₂ = 1.
    let a_bandec: Vec<Vec<f64>> = vec![
        // Row 0: only the diagonal and superdiagonal elements are stored.
        vec![0.0, 2.0, 1.0],
        // Row 1: both sub- and superdiagonals are stored.
        vec![1.0, 2.0, 1.0],
        // Row 2: only the subdiagonal and diagonal elements are stored.
        vec![1.0, 2.0, 0.0],
    ];
    let m1 = 1;
    let m2 = 1;
    let bandec = Bandec::new(&a_bandec, m1, m2);

    // Let b be the right-hand side vector.
    let b = vec![3.0, 4.0, 3.0];
    let mut x = vec![0.0; 3];
    bandec.solve(&b, &mut x);
    println!("Solution from Bandec::solve, x = {:?}", x);
    println!("Determinant of A = {:.5}", bandec.det());
}