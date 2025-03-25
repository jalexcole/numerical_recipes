use std::slice;

use num_complex::{Complex, ComplexFloat};

pub fn laguer(a: &[Complex<f64>], x: &mut Complex<f64>, its: &mut usize) -> Complex<f64>{
    let mut temp_x = x.clone();
    const MR: usize = 8;
    const MT: usize = 10;
    const MAXIT: usize = MT * MR;
    const EPS: f64 = f64::EPSILON;
    const FRAC: [f64; MR + 1] = [0.0,0.5,0.25,0.75,0.13,0.38,0.62,0.88,1.0];
    let mut dx;
    let mut x1: Complex<f64>;
    let mut b;
    let mut d: Complex<f64>;
    let mut f: Complex<f64>;
    let mut g;
    let mut h;
    let mut sq;
    let mut gp;
    let mut gm;
    let mut g2;
    let m = a.len() - 1;

    for iter in 1..=MAXIT {
        *its = iter;
        b = a[m];
        let mut err = b.abs();
        f = Complex::ZERO;
        d = f;
        let abx = temp_x.abs();
        (m - 1..=0).for_each(|j| {
            f = temp_x * f + d;
            //
            d = temp_x * d + b;
            b = temp_x * b + a[j];
            err = b.abs() + abx * err;
        });
        err *= EPS;
        // Estimate of roundoﬀ error in evaluating polynomial.
        if b.abs() <= err {
            *x =  temp_x;
            return temp_x;
        }
        g = d / b; // We are on the root.
        // The generic case: Use Laguerre’s formula.
        g2 = g * g;
        h = g2 - 2.0 * f / b;
        sq = ((m - 1) as f64 * ((m as f64) * h - g2)).sqrt();
        gp = g + sq;
        gm = g - sq;
        let abp = (gp).abs();
        let abm = (gm).abs();
        if abp < abm {
            gp = gm;
        }
        dx = match f64::max(abp, abm) > 0.0 {
            true => Complex::new(m as f64, 0.0) / gp,
            false => Complex::from_polar(1.0 + abx, iter as f64),
        };
        x1 = temp_x - dx;
        if temp_x == x1 {
            *x =  temp_x;
            return temp_x;
        } // Converged.
        if iter % MT != 0 {
            temp_x =  x1;
        } else {
            temp_x -= Complex::new(FRAC[iter / MT], 0.0) * dx;
        }
        // Every so often we take a fractional step, to break any limit cycle (itself a rare occur-
        // rence).
    }
    panic!("too many iterations in laguer");
}

/// Given the m+1 complex coeﬃcients a[0..m] of the polynomial
/// $$\sum^m_{i=0} a(i) x^i$$
/// this routine successively calls laguer and finds all m complex roots in roots[0..m-1]. The boolean variable
/// polish should be input as true if polishing (also by Laguerre’s method) is desired, false if the
/// roots will be subsequently polished by other means.
pub fn zroots<T>(a: &[Complex<f64>], roots: &mut [Complex<f64>], polish: bool) {
    const EPS: f64 = 1.0e-14;
    let i: usize = 0;
    let mut its=0;

    let mut x;
    let mut b;
    let mut c;

    let m = a.len() - 1;

    let mut ad ;// = vec![Complex::new(0.0, 0.0); m + 1];

    ad = a.to_vec();
    

    for j in m - 1..=0 {
        x = Complex::default();
        let mut ad_v = vec![Complex::new(0.0, 0.0); j + 2];
        #[allow(clippy::manual_memcpy)]
        for jj in 0..j + 2 {
            ad_v[jj] = ad[jj];
        }
        laguer(&ad_v, &mut x, &mut its);
        if (x.im().abs() <= 2.0 * EPS * x.re().abs()) {
            x = Complex::new(x.re(), 0.0);
        }
        roots[j] = x;
        b = ad[j + 1];
        for jj in j..=0 {
            c = ad[jj];
            ad[jj] = b;
            b = x * b + c;
        }
    }

    if polish {
        (0..m).for_each(|j| {
            roots[j] = laguer(a, &mut roots[j], &mut its);
        });
    }

    for j in 1..m {
        x = roots[j];
        for i in j - 1..=0 {
            if roots[i].re() <= x.re() {
                break;
            }
            roots[i + 1] = roots[i];
        }
        roots[i + 1] = x;
    }
}
