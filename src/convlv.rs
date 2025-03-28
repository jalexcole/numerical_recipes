use crate::fourier::realft;

/// Convolves or deconvolves a real data set data[0..n-1] (including any user-supplied zero padding) with a response function response[0..m-1], where m is an odd integer <= n. The response function must be stored in wraparound order: The first half of the array respns contains the impulse response function at positive times, while the second half of the array contains the impulse response function at negative times, counting down from the highest element respons[m-1]. On input isign is C1 for convolution, 1 for deconvolution. The answer is returned in ans[0..n-1]. n must be an integer power of 2.
pub fn convlv(data: &[f64], respns: &[f64], isign: isize, ans: &mut [f64]) {
    let no2;
    let n = data.len();
    let m = respns.len();

    let mut mag2;
    let mut tmp;

    let mut temp = vec![0.0; n];
    temp[0] = respns[0];
    for i in 1..(m + 1) / 2 {
        temp[i] = respns[i];
        temp[n - i] = respns[m - i];
        // Put respns in array of length n.
    }
    for i in m + 1 / 2..n - (m - 1) / 2
    // Pad with zeros.
    {
        temp[i] = 0.0;
    }
    for i in 0..n {
        ans[i] = data[i];
    }
    realft(ans, 1); // FFT both arrays.
    realft(&mut temp, 1);
    no2 = n >> 1;
    if (isign == 1) {
        for i in (2..n).step_by(2) {
            // Multiply FFTs to convolve.
            tmp = ans[i];
            ans[i] = (ans[i] * temp[i] - ans[i + 1] * temp[i + 1]) / no2 as f64;
            ans[i + 1] = (ans[i + 1] * temp[i] + tmp * temp[i + 1]) / no2 as f64;
        }
        ans[0] = ans[0] * temp[0] / no2 as f64;
        ans[1] = ans[1] * temp[1] / no2 as f64;
    } else if (isign == -1) {
        for i in (2..n).step_by(2) {
            // Divide FFTs to deconvolve.
            mag2 = temp[i].powf(2.0) + temp[i + 1].powf(2.0);
            if mag2 == 0.0 {
                panic!("Deconvolving at response zero in convlv");
            }
            tmp = ans[i];
            ans[i] = (ans[i] * temp[i] + ans[i + 1] * temp[i + 1]) / mag2 / no2 as f64;
            ans[i + 1] = (ans[i + 1] * temp[i] - tmp * temp[i + 1]) / mag2 / no2 as f64;
        }
        if (temp[0] == 0.0 || temp[1] == 0.0) {
            panic!("Deconvolving at response zero in convlv");
        }
        ans[0] = ans[0] / temp[0] / no2 as f64;
        ans[1] = ans[1] / temp[1] / no2 as f64;
    } else {
        panic!("No meaning for isign in convlv");
    }
    realft(ans, -1); // Inverse transform back to time domain.
}
