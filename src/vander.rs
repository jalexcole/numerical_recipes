/// Solves the Vandermonde linear system:
/// 
///     ∑ (x[i]^k * w[i]) = q[k]   for k = 0, 1, …, n-1,
/// 
/// where the input vectors are `x[0..n]` and `q[0..n]`, and the output vector `w[0..n]` is computed.
/// 
/// # Panics
/// 
/// Panics if the lengths of the slices are inconsistent.
pub fn vander(x: &[f64], w: &mut [f64], q: &[f64]) {
    let n = q.len();
    assert_eq!(x.len(), n, "x and q must have the same length");
    assert_eq!(w.len(), n, "w must have length equal to q");
    
    // Create a temporary workspace vector c of length n.
    let mut c = vec![0.0; n];

    if n == 1 {
        w[0] = q[0];
    } else {
        // Initialize c to all zeros.
        // c[n-1] is set to -x[0] (the coefficient for the master polynomial).
        c[n - 1] = -x[0];
        // For each x[i] for i = 1 to n-1, update the coefficients by recursion.
        for i in 1..n {
            let xx = -x[i];
            // Update coefficients from index (n-1-i) up to n-2.
            for j in (n - 1 - i)..(n - 1) {
                c[j] += xx * c[j + 1];
            }
            // Update the last coefficient.
            c[n - 1] += xx;
        }
        // Now, for each component of the solution vector w:
        for i in 0..n {
            let xx = x[i];
            let mut t = 1.0;
            let mut b_val = 1.0;
            let mut s = q[n - 1];
            // Loop backwards: k from n-1 downto 1.
            for k in (1..n).rev() {
                b_val = c[k] + xx * b_val;
                s += q[k - 1] * b_val;
                t = xx * t + b_val;
            }
            w[i] = s / t;
        }
    }
}

fn main() {
    // Example usage:
    // Suppose we wish to solve for w in the Vandermonde system:
    //    sum_{i=0}^{n-1} (x[i]^k * w[i]) = q[k] for k = 0, …, n-1.
    //
    // For a simple example, let n = 3:
    //     x = [2.0, 3.0, 5.0]
    //     q = [1.0, 4.0, 9.0]
    //
    // Then the function computes the corresponding vector w.
    let x = vec![2.0, 3.0, 5.0];
    let q = vec![1.0, 4.0, 9.0];
    let mut w = vec![0.0; 3];

    vander(&x, &mut w, &q);

    println!("Solution w: {:?}", w);
}