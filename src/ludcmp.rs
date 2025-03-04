use crate::nr3::NRmatrix;

/// LU Decomposition object for solving linear equations and related operations.
/// ```rust
/// let mut a = NRmatrix::with_value(3, 3, 1.0);
/// *a.get_mut(0, 0) = 2.0;
/// *a.get_mut(0, 1) = -1.0;
/// *a.get_mut(0, 2) = 0.0;
/// *a.get_mut(1, 0) = -1.0;
/// *a.get_mut(1, 1) = 2.0;
/// *a.get_mut(1, 2) = -1.0;
/// *a.get_mut(2, 0) = 0.0;
/// *a.get_mut(2, 1) = -1.0;
/// *a.get_mut(2, 2) = 2.0;
///
/// let b = vec![1.0, 2.0, 3.0];
/// let mut x = vec![0.0; 3];
///
/// let lu = LUdcmp::new(&a);
/// lu.solve(&b, &mut x);

/// println!("Solution vector x:");
/// for xi in &x {
///     println!("{:.3}", xi);
/// }```
pub struct LUdcmp {
    n: usize,
    lu: NRmatrix<f64>,
    indx: Vec<usize>,
    d: f64,
}

impl LUdcmp {
    /// Constructor. Takes a matrix `a` and computes its LU decomposition.
    pub fn new(a: &NRmatrix<f64>) -> Self {
        let n = a.nrows();
        let mut lu = a.clone();
        let mut indx = vec![0; n];
        let mut d = 1.0;
        let mut vv = vec![0.0; n];
        const TINY: f64 = 1.0e-40;

        for i in 0..n {
            let mut big = 0.0;
            for j in 0..n {
                let temp = lu.get(i, j).abs();
                if temp > big {
                    big = temp;
                }
            }
            if big == 0.0 {
                panic!("Singular matrix in LUdcmp");
            }
            vv[i] = 1.0 / big;
        }

        for k in 0..n {
            let mut big = 0.0;
            let mut imax = k;

            for i in k..n {
                let temp = vv[i] * lu.get(i, k).abs();
                if temp > big {
                    big = temp;
                    imax = i;
                }
            }

            if k != imax {
                lu.swap_rows(imax, k);
                d = -d;
                vv.swap(imax, k);
            }

            indx[k] = imax;
            if *lu.get(k, k) == 0.0 {
                *lu.get_mut(k, k) = TINY;
            }

            for i in k + 1..n {
                let temp = *lu.get_mut(i, k) / *lu.get(k, k);
                for j in k + 1..n {
                    *lu.get_mut(i, j) -= temp * *lu.get(k, j);
                }
            }
        }

        LUdcmp { n, lu: lu.clone(), indx, d }
    }

    /// Solves for a single right-hand side.
    pub fn solve(&self, b: &[f64], x: &mut [f64]) {
        if b.len() != self.n || x.len() != self.n {
            panic!("LUdcmp::solve bad sizes");
        }

        for i in 0..self.n {
            x[i] = b[i];
        }

        let mut ii = 0;
        for i in 0..self.n {
            let ip = self.indx[i];
            let mut sum = x[ip];
            x[ip] = x[i];
            if ii != 0 {
                for j in 0..i {
                    sum -= self.lu.get(i, j) * x[j];
                }
            } else if sum != 0.0 {
                ii = i + 1;
            }
            x[i] = sum;
        }

        for i in (0..self.n).rev() {
            let mut sum = x[i];
            for j in i + 1..self.n {
                sum -= self.lu.get(i, j) * x[j];
            }
            x[i] = sum / self.lu.get(i, i);
        }
    }

    /// Solves for multiple right-hand sides.
    pub fn solve_matrix(&self, b: &NRmatrix<f64>, x: &mut NRmatrix<f64>) {
        let m = b.ncols();
        if b.nrows() != self.n || x.nrows() != self.n || b.ncols() != x.ncols() {
            panic!("LUdcmp::solve_matrix bad sizes");
        }

        let mut xx = vec![0.0; self.n];
        for j in 0..m {
            for i in 0..self.n {
                xx[i] = *b.get(i, j);
            }
            self.solve(&xx, &mut xx.clone());
            for i in 0..self.n {
                *x.get_mut(i, j) = xx[i];
            }
        }
    }
}
