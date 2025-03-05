/// Solves the tridiagonal system of equations:
/// 
/// a[i] * u[i-1] + b[i] * u[i] + c[i] * u[i+1] = r[i]  for i = 0,1,...,n-1,
/// 
/// where:
/// - `a[0..n-1]`, `b[0..n-1]`, `c[0..n-1]`, and `r[0..n-1]` are input vectors.
///   (Note: a[0] and c[n-1] are not used.)
/// - `u[0..n-1]` is the output solution vector.
/// 
/// # Arguments
/// * `a` - Sub-diagonal coefficients (length n; a[0] is unused).
/// * `b` - Diagonal coefficients (length n).
/// * `c` - Super-diagonal coefficients (length n; c[n-1] is unused).
/// * `r` - Right-hand side vector (length n).
/// * `u` - Output vector (length n) that will contain the solution.
/// 
/// # Panics
/// Panics if b[0] is zero ("Error 1 in tridag") or if any computed beta equals zero ("Error 2 in tridag").
pub fn tridag(a: &[f64], b: &[f64], c: &[f64], r: &[f64], u: &mut [f64]) {
    let n = a.len();
    // Check that all input slices have the same length.
    assert_eq!(b.len(), n, "Length of b must equal length of a");
    assert_eq!(c.len(), n, "Length of c must equal length of a");
    assert_eq!(r.len(), n, "Length of r must equal length of a");
    assert_eq!(u.len(), n, "Length of u must equal length of a");

    let mut gam = vec![0.0; n];

    if b[0] == 0.0 {
        panic!("Error 1 in tridag");
    }
    let mut bet = b[0];
    u[0] = r[0] / bet;
    for j in 1..n {
        gam[j] = c[j - 1] / bet;
        bet = b[j] - a[j] * gam[j];
        if bet == 0.0 {
            panic!("Error 2 in tridag");
        }
        u[j] = (r[j] - a[j] * u[j - 1]) / bet;
    }
    // Backsubstitution loop: j from n-2 downto 0.
    for j in (0..(n - 1)).rev() {
        u[j] -= gam[j + 1] * u[j + 1];
    }
}

fn main() {
    // Example:
    // Solve the tridiagonal system:
    //
    //   b[0]*u[0] + c[0]*u[1]           = r[0]
    //   a[1]*u[0] + b[1]*u[1] + c[1]*u[2] = r[1]
    //   a[2]*u[1] + b[2]*u[2] + c[2]*u[3] = r[2]
    //   a[3]*u[2] + b[3]*u[3]           = r[3]
    //
    // For this example, we choose:
    //   a = [0, 1, 1, 1]    (a[0] is unused)
    //   b = [2, 2, 2, 2]
    //   c = [1, 1, 1, 0]    (c[3] is unused)
    //   r = [3, 4, 4, 3]
    //
    // The expected solution is u = [1, 1, 1, 1].
    let a = vec![0.0, 1.0, 1.0, 1.0];
    let b = vec![2.0, 2.0, 2.0, 2.0];
    let c = vec![1.0, 1.0, 1.0, 0.0];
    let r = vec![3.0, 4.0, 4.0, 3.0];
    let mut u = vec![0.0; 4];

    tridag(&a, &b, &c, &r, &mut u);

    println!("Solution u: {:?}", u);
}