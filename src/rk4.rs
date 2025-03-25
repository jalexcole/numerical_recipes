pub fn rk4(
    y: &[f64],
    dydx: &[f64],
    x: f64,
    h: f64,
    yout: &mut [f64],
    derivs: impl Fn(f64, &[f64], &mut [f64]),
) {
    let n = y.len();

    let mut dym = vec![0.0; n];
    let mut dyt = vec![0.0; n];
    let mut yt = vec![0.0; n];

    let hh = h * 0.5;
    let h6 = h / 6.0;
    let xh = x + hh;

    for i in 0..n {
        yt[i] = y[i] + hh * dydx[i];
    }

    derivs(xh, &yt, &mut dyt);

    for i in 0..n {
        yt[i] = y[i] + hh * dyt[i];
    }

    derivs(xh, &yt, &mut dym);

    for i in 0..n {
        yt[i] = y[i] + h * dym[i];
        dym[i] += dyt[i];
    }

    derivs(x + h, &yt, &mut dyt);

    for i in 0..n {
        yout[i] = y[i] + h6 * (dydx[i] + dyt[i] + 2.0 * dym[i]);
    }
}
