use crate::nr3::NRmatrix;

/// A matrix structure for numerical operations using a single vector for data storage.
///
/// # Example
/// ```
/// use numerical_recipes::nr3::NRmatrix;
/// use numerical_recipes::guassj::gaussj;
///
/// let mut a = NRmatrix::with_value(3, 3, 1.0);
///  *a.get_mut(0, 0) = 2.0;
///  *a.get_mut(0, 1) = -1.0;
///  *a.get_mut(0, 2) = 0.0;
///  *a.get_mut(1, 0) = -1.0;
///  *a.get_mut(1, 1) = 2.0;
///  *a.get_mut(1, 2) = -1.0;
///  *a.get_mut(2, 0) = 0.0;
///  *a.get_mut(2, 1) = -1.0;
///  *a.get_mut(2, 2) = 2.0;
///
///  let mut b = NRmatrix::with_value(3, 1, 0.0);
///   *b.get_mut(0, 0) = 1.0;
///   *b.get_mut(1, 0) = 2.0;
///   *b.get_mut(2, 0) = 3.0;
///
///   gaussj(&mut a, &mut b);
///
///   println!("Inverted matrix a:");
///   for i in 0..a.nrows() {
///      for j in 0..a.ncols() {
///         print!("{:.3} ", a.get(i, j));
///       }
///       println!();
///     }
///
///     println!("Solution matrix b:");
///     for i in 0..b.nrows() {
///         for j in 0..b.ncols() {
///             print!("{:.3} ", b.get(i, j));
///         }
///         println!();
///     }
///
/// ```
pub fn gaussj(a: &mut NRmatrix<f64>, b: &mut NRmatrix<f64>) {
    let n = a.nrows();
    let m = b.ncols();

    let mut indxc = vec![0; n];
    let mut indxr = vec![0; n];
    let mut ipiv = vec![0; n];

    for i in 0..n {
        let mut big = 0.0;
        let mut irow = 0;
        let mut icol = 0;

        for j in 0..n {
            if ipiv[j] != 1 {
                for k in 0..n {
                    if ipiv[k] == 0 {
                        let abs_a = a.get(j, k).abs();
                        if abs_a >= big {
                            big = abs_a;
                            irow = j;
                            icol = k;
                        }
                    }
                }
            }
        }

        ipiv[icol] += 1;

        if irow != icol {
            a.swap_rows(irow, icol);
            b.swap_rows(irow, icol);
        }

        indxr[i] = irow;
        indxc[i] = icol;

        if *a.get(icol, icol) == 0.0 {
            panic!("gaussj: Singular Matrix");
        }

        let pivinv = 1.0 / *a.get(icol, icol);
        *a.get_mut(icol, icol) = 1.0;

        for l in 0..n {
            *a.get_mut(icol, l) *= pivinv;
        }

        for l in 0..m {
            *b.get_mut(icol, l) *= pivinv;
        }

        for ll in 0..n {
            if ll != icol {
                let dum = *a.get(ll, icol);
                *a.get_mut(ll, icol) = 0.0;

                for l in 0..n {
                    *a.get_mut(ll, l) -= *a.get(icol, l) * dum;
                }

                for l in 0..m {
                    *b.get_mut(ll, l) -= *b.get(icol, l) * dum;
                }
            }
        }
    }

    for l in (0..n).rev() {
        if indxr[l] != indxc[l] {
            a.swap_cols(indxr[l], indxc[l]);
        }
    }
}
