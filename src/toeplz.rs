/// Solves the Toeplitz system
/// 
///     for 0 ≤ i ≤ n-1:   ∑₍ⱼ₌₀₎ⁿ⁻¹  r[n-1 + i - j] · x[j] = y[i],
/// 
/// where:
///   - `r` is a vector of length 2*n - 1 describing the Toeplitz matrix (its diagonal is at index n-1),
///   - `y` is the right-hand side (length n),
///   - `x` is the output vector (length n).
/// 
/// # Panics
/// 
/// Panics if a singular principal minor is encountered.
pub fn toeplz(r: &[f64], x: &mut [f64], y: &[f64]) {
    // Let n be the size of the system (from y and x)
    let n = y.len();
    // r is assumed to have length 2*n - 1.
    let n1 = n - 1; // principal index for the diagonal of the Toeplitz matrix

    // Check first principal minor
    if r[n1] == 0.0 {
        panic!("toeplz-1 singular principal minor");
    }
    // Initialize the first solution component.
    x[0] = y[0] / r[n1];
    // If the system is 1×1, we are done.
    if n1 == 0 {
        return;
    }

    // Workspace vectors g and h of length n1.
    // (In the original code these are allocated with size n-1.)
    let mut g = vec![0.0; n1];
    let mut h = vec![0.0; n1];

    // Initialize g[0] and h[0]
    g[0] = r[n1 - 1] / r[n1];
    h[0] = r[n1 + 1] / r[n1];

    // Main recursion loop.
    // The loop variable m runs from 0 to n-2 (so that m+1 is at most n-1).
    for m in 0..(n - 1) {
        let m1 = m + 1;
        // Compute numerator (sxn) and denominator (sd) for x[m1]
        let mut sxn = -y[m1];
        let mut sd = -r[n1];
        for j in 0..(m + 1) {
            // r index: n1 + m1 - j   (must lie in  [n1, 2*n - 2])
            sxn += r[n1 + m1 - j] * x[j];
            // g index: m - j, with m - j in 0..=m (m is at most n-2 so this is safe)
            sd += r[n1 + m1 - j] * g[m - j];
        }
        if sd == 0.0 {
            panic!("toeplz-2 singular principal minor");
        }
        x[m1] = sxn / sd;
        // Update previously computed x values.
        for j in 0..(m + 1) {
            x[j] -= x[m1] * g[m - j];
        }
        // If we have reached the last row, return.
        if m1 == n1 {
            return;
        }
        // Compute new values for g and h.
        let mut sgn = -r[n1 - m1 - 1];
        let mut shn = -r[n1 + m1 + 1];
        let mut sgd = -r[n1];
        for j in 0..(m + 1) {
            sgn += r[n1 + j - m1] * g[j];
            shn += r[n1 + m1 - j] * h[j];
            sgd += r[n1 + j - m1] * h[m - j];
        }
        if sgd == 0.0 {
            panic!("toeplz-3 singular principal minor");
        }
        // Since g and h have length n1 (indices 0..n1-1),
        // m1 must be less than n1; note that the loop will return before m1 equals n1.
        g[m1] = sgn / sgd;
        h[m1] = shn / sd;

        // Update the remaining coefficients in g and h by a symmetric correction.
        let mut k = m; // starting index for reverse sweep
        // m2 is essentially floor((m+2)/2)
        let m2 = (m + 2) >> 1;
        let pp = g[m1];
        let qq = h[m1];
        for _j in 0..m2 {
            let pt1 = g[_j];
            let pt2 = g[k];
            let qt1 = h[_j];
            let qt2 = h[k];
            g[_j] = pt1 - pp * qt2;
            g[k] = pt2 - pp * qt1;
            h[_j] = qt1 - qq * pt2;
            h[k] = qt2 - qq * pt1;
            if k == 0 {
                break;
            }
            k -= 1;
        }
    }
    panic!("toeplz - should not arrive here!");
}

fn main() {
    // --- Example Usage ---
    //
    // Suppose we want to solve the Toeplitz system
    //   ∑₍ⱼ₌₀₎ⁿ⁻¹ r[n-1 + i - j] · x[j] = y[i]   for i = 0, 1, …, n-1.
    //
    // For example, let n = 4. Then r must have length 2*4 - 1 = 7.
    // Here we choose an r that defines the Toeplitz matrix (its diagonal is at index 3),
    // a right-hand side vector y, and we will compute the solution x.
    let r = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]; // length 7; diagonal is r[3] = 4.0
    let y = vec![2.0, 3.0, 4.0, 5.0]; // length 4
    let mut x = vec![0.0; 4];         // solution vector of length 4

    // Call the solver. (If a singular principal minor is detected, this will panic.)
    toeplz(&r, &mut x, &y);

    println!("Solution x: {:?}", x);
}