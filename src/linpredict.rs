#![cfg_attr(feature = "nightly", feature(portable_simd))]

/// Given a real vector of `data[0..n-1]`, this routine returns m linear
/// prediction coeﬃcients as `d[0..m-1]`, and returns the mean square
/// discrepancy as `xms``.
#[cfg(feature = "default")]
pub fn memcof(data: &[f64], xms: &mut f64, d: &mut [f64]) {
    let n = data.len();
    let m = d.len();
    let mut p = 0.0;

    let mut wk1 = vec![0.0; n];
    let mut wk2 = vec![0.0; n];
    let mut wkm = vec![0.0; m];

    for j in data.iter() {
        p += j * j;
    }

    *xms = p / n as f64;
    wk1[0] = data[0];
    wk2[n - 2] = data[n - 1];

    #[allow(clippy::manual_memcpy)]
    for j in 1..n - 1 {
        wk1[j] = data[j];
        wk2[j - 1] = data[j];
    }
    for k in 0..m {
        let mut num = 0.0;
        let mut denom = 0.0;

        for j in 0..n - k - 1 {
            num += wk1[j] * wk2[j];
            denom += (wk1[j].powf(2.0)) + wk2[j].powf(2.0);
        }
        d[k] = 2.0 * num / denom;
        *xms *= 1.0 - (d[k].powf(2.0));
        for i in 0..k {
            d[i] = wkm[i] - d[k] * wkm[k - 1 - i];
            // The algorithm is recursive, building up the answer for larger and larger values of m
            // until the desired value is reached. At this point in the algorithm, one could return
            // the vector d and scalar xms for a set of LP coeﬃcients with k (rather than m)
            // terms.
        }
        if k == m - 1 {
            return;
        }
        #[allow(clippy::manual_memcpy)]
        for i in 0..=k {
            wkm[i] = d[i];
        }
        for j in 0..n - k - 2 {
            wk1[j] -= wkm[k] * wk2[j];
            wk2[j] = wk2[j + 1] - wkm[k] * wk1[j + 1];
        }
    }
    panic!("Never get here in memcof");
}

#[cfg(feature = "nightly")]
use std::simd::Simd;
#[cfg(feature = "nightly")]
use std::simd::num::SimdFloat;

use num_complex::{Complex, ComplexFloat};
#[cfg(feature = "nightly")]
pub fn memcof(data: &[f64], xms: &mut f64, d: &mut [f64]) {
    let n = data.len();
    let m = d.len();

    let mut wk1 = vec![0.0; n];
    let mut wk2 = vec![0.0; n];
    let mut wkm = vec![0.0; m];

    // SIMD reduction for sum of squares.
    const LANES: usize = 8;
    let mut sum = Simd::<f64, LANES>::splat(0.0);
    let chunks = data.chunks_exact(LANES);
    let remainder = chunks.remainder();
    for chunk in chunks {
        let vec = Simd::<f64, LANES>::from_slice(chunk);
        sum += vec * vec;
    }
    let mut p = sum.reduce_sum();
    for &val in remainder {
        p += val * val;
    }

    *xms = p / n as f64;
    wk1[0] = data[0];
    wk2[n - 2] = data[n - 1];

    // Copy data into wk1 and wk2.
    for j in 1..n - 1 {
        wk1[j] = data[j];
        wk2[j - 1] = data[j];
    }

    for k in 0..m {
        let mut num = 0.0;
        let mut denom = 0.0;
        let len = n - k - 1;
        let mut j = 0;
        while j + LANES <= len {
            let v1 = Simd::<f64, LANES>::from_slice(&wk1[j..j + LANES]);
            let v2 = Simd::<f64, LANES>::from_slice(&wk2[j..j + LANES]);
            num += (v1 * v2).reduce_sum();
            denom += (v1 * v1 + v2 * v2).reduce_sum();
            j += LANES;
        }
        for j in j..len {
            num += wk1[j] * wk2[j];
            denom += wk1[j] * wk1[j] + wk2[j] * wk2[j];
        }
        d[k] = 2.0 * num / denom;
        *xms *= 1.0 - d[k] * d[k];
        for i in 0..k {
            d[i] = wkm[i] - d[k] * wkm[k - 1 - i];
        }
        if k == m - 1 {
            return;
        }
        // Update wkm with current coefficients.
        for i in 0..=k {
            wkm[i] = d[i];
        }
        // Update wk1 and wk2.
        let len = n - k - 2;
        let mut j = 0;
        while j + LANES <= len {
            let mut v_wk1 = Simd::<f64, LANES>::from_slice(&wk1[j..j + LANES]);
            let v_wk2 = Simd::<f64, LANES>::from_slice(&wk2[j..j + LANES]);
            let coeff = Simd::<f64, LANES>::splat(wkm[k]);
            let new_wk1 = v_wk1 - coeff * v_wk2;
            new_wk1.write_to_slice(&mut wk1[j..j + LANES]);
            // Note: The update for wk2 involves a shift (accessing wk2[j+1] and wk1[j+1]),
            // which may be less straightforward to vectorize.
            j += LANES;
        }
        for j in j..len {
            wk1[j] -= wkm[k] * wk2[j];
            wk2[j] = wk2[j + 1] - wkm[k] * wk1[j + 1];
        }
    }
}

/// Given the LP coeﬃcients d[0..m-1], this routine finds all roots of the characteristic polynomial
/// (13.6.14), reflects any roots that are outside the unit circle back inside, and then returns a
/// modified set of coeﬃcients d[0..m-1].
pub fn fixrts(d: &mut [f64]) {
    let mut polish = true;
    let m = d.len();

    let mut a = vec![num_complex::Complex::new(0.0, 0.0); m + 1];
    let mut roots = vec![num_complex::Complex::new(0.0, 0.0); m + 1];

    for j in 0..m {
        a[j] = Complex::new(-1.0 * d[m - 1 - j], 0.0);
    }

    zroots(a, roots, polish);
    for j in 0..m {
        if roots[j].abs() > 1.0 {
            roots[j] = -1.0 / (roots[j]).conj();
        }
    }

    a[0] = -roots[0]; // Now reconstruct the polynomial coeﬃcients,
    a[1] = Complex::new(1.0, 0.0);
    for j in 1..m {
        // by looping over the roots
        a[j + 1] = Complex::new(1.0, 0.0);
        for i in j..=1 {
            // and synthetically multiplying.
            a[i] = a[i - 1] - roots[j] * a[i];
            a[0] = -roots[j] * a[0];
        }
        for j in 0..m {
            d[m - 1 - j] = -a[j].re();
        }
    }
}
/// Given data[0..ndata-1], and given the data’s LP coeﬃcients d[0..m-1], this routine applies equation (13.6.11) to predict the next nfut data points, which it returns in the array future[0..nfut-1]. Note that the routine references only the last m values of data, as initial values for the prediction.
pub fn predic(data: &[f64], d: &[f64], future: &mut [f64]) {
    let ndata = data.len();
    let m = d.len();
    let nfut = future.len();
    let mut sum;
    let mut discrp;
    let mut reg = vec![0.0; m];
    for j in 0..m {
        reg[j] = data[ndata - 1 - j];
    }
    for j in 0..nfut {
        discrp = 0.0;
        // This is where you would put in a known discrepancy if you were reconstructing a func-
        // tion by linear predictive coding rather than extrapolating a function by linear prediction.
        // See text.
        sum = discrp;
        for k in 0..m {
            sum += d[k] * reg[k];
        }
        for k in m - 1..=1 {
            reg[k] = reg[k - 1];
        }
        reg[0] = sum;
        future[j] = reg[0];
    }
    // [If you want to implement circulararrays, you can avoid this shift-ing of coeﬃcients.]
}

pub fn evlmem(fdt: f64, d: &[f64], xms: f64) -> f64 {
    let mut sumr = 1.0;
    let mut sumi = 0.0;
    let mut wr = 1.0;
    let mut wi = 0.0;
    let mut wtemp;
    let m = d.len();
    let theta = 6.28318530717959 * fdt;
    let wpr = theta.cos();
    let wpi = theta.sin();

    for i in 0..m {
        wtemp = wr;
        wr = wr * wpr - wi * wpi;
        wi = wi * wpr + wtemp * wpi;
        sumr -= d[i] * wr;
        sumi -= d[i] * wi;
    }
    return xms / (sumr * sumr + sumi * sumi);
}
