use crate::fourier::realft;

/// Computes the correlation of two real data sets data1[0..n-1] and data2[0..n-1] (including
/// any user-supplied zero padding). n must be an integer power of 2. The answer is returned in
/// ans[0..n-1] stored in wraparound order, i.e., correlations at increasingly negative lags are in
/// ans[n-1] on down to ans[n/2], while correlations at increasingly positive lags are in ans[0]
/// (zero lag) on up to ans[n/2-1]. Sign convention of this routine: if data1 lags data2, i.e., is
/// shifted to the right of it, then ans will show a peak at positive lags.
pub fn correl(data1: &[f64], data2: &[f64], ans: &mut [f64]) {
    let no2;
    let n = data1.len();
    let mut tmp;
    let mut temp = vec![0.0; n];

    for i in 0..n {
        ans[i] = data1[i];
        temp[i] = data2[i];
    }

    realft(ans, 1);
    realft(&mut temp, 1);

    no2 = n >> 1;
    for i in (2..n).step_by(2) {
        tmp = ans[i];
        ans[i] = (ans[i] * temp[i] + ans[i + 1] * temp[i + 1]) / no2 as f64;
        ans[i + 1] = (ans[i + 1] * temp[i] - tmp * temp[i + 1]) / no2 as f64;
    }

    ans[0] = ans[0] * temp[0] / no2 as f64;
    ans[1] = ans[1] * temp[1] / no2 as f64;
    realft(ans, -1);
}
