use std::f64::consts::{PI, TAU};

pub fn four1(data: &mut [f64], isign: isize) {
    let n = data.len();
    let mut mmax;
    let mut m;
    let mut j;
    let mut istep;

    let mut wtemp;
    let mut wr;
    let mut wpr;
    let mut wpi;
    let mut wi;
    let mut theta;
    let mut tempr;
    let mut tempi;

    if n < 2 || n & (n - 1) != 0 {
        panic!("Four1: n must be a power of 2");
    }

    let nn = n << 1;
    j = 1;

    for i in (1..nn).step_by(2) {
        if j > i {
            data.swap(j - 1, i - 1);
            data.swap(j, i);
        }

        m = n;
        while m >= 2 && j > m {
            j -= m;
            m >>= 1;
        }

        j += m;
    }

    mmax = 2;
    while nn > mmax {
        istep = mmax << 1;
        theta = isign as f64 * TAU / mmax as f64;
        wtemp = (theta * 0.5).sin();
        wpr = -2.0 * wtemp * wtemp;
        wpi = theta.sin();
        wr = 1.0;
        wi = 0.0;

        for _ in (1..mmax).step_by(2) {
            for i in (1..mmax).step_by(istep) {
                j = i + mmax;
                tempr = wr * data[j - 1] - wi * data[j];
                tempi = wr * data[j] + wi * data[j - 1];
                data[j - 1] = data[i - 1] - tempr;
                data[j] = data[i] - tempi;
                data[i - 1] += tempr;
                data[i] += tempi;
            }

            wr = (wpr * wr) - (wpi * wi) + wr;
            wi = (wpr * wi) + (wpi * wr) + wi;
        }

        mmax = istep;
    }
}

pub fn realft(data: &mut [f64], isign: isize) {
    let mut i1;
    let mut i2;
    let mut i3;
    let mut i4;
    let n = data.len();
    let c1 = 0.5;
    let c2;
    let mut h1r;
    let mut h1i;
    let mut h2r;
    let mut h2i;
    let mut wr;
    let mut wi;

    let mut wtemp;
    let mut theta = std::f64::consts::PI / (n >> 1) as f64; // Initialize the recurrence.
    if isign == 1 {
        c2 = -0.5;
        four1(data, 1); // The forward transform is here.
    } else {
        c2 = 0.5;
        theta = -theta; //Otherwise set up for an inverse transform.
    }
    wtemp = f64::sin(0.5 * theta);
    let wpr = -2.0 * wtemp * wtemp;
    let wpi = theta.sin();
    wr = 1.0 + wpr;
    wi = wpi;

    for i in 1..(n >> 2) {
        // Case i=0 done separately below.
        i1 = i + i;
        i2 = 1 + i1;
        i3 = n - i1;
        i4 = 1 + i3;
        h1r = c1 * (data[i1] + data[i3]); // The two separate transforms are separated out of data.
        h1i = c1 * (data[i2] - data[i4]);
        h2r = -c2 * (data[i2] + data[i4]);
        h2i = c2 * (data[i1] - data[i3]);
        data[i1] = h1r + wr * h2r - wi * h2i; // Here they are recombined to form the true transform of the original real data.
        data[i2] = h1i + wr * h2i + wi * h2r;
        data[i3] = h1r - wr * h2r + wi * h2i;
        data[i4] = -h1i + wr * h2i + wi * h2r;
        wtemp = wr;
        wr += wtemp * wpr - wi * wpi; // The recurrence.
        wi = wi * wpr + wtemp * wpi + wi;
    }
    if isign == 1 {
        h1r = data[0];
        data[0] = (h1r) + data[1]; // Squeeze the first and last data together to get them all within the original array.
        data[1] = h1r - data[1];
    } else {
        h1r = data[0];
        data[0] = c1 * (h1r + data[1]);
        data[1] = c1 * (h1r - data[1]);
        four1(data, -1); // This is the inverse transform for the case `isign=-1`.
    }
}

/// Calculates the sine transform of a set of n real-valued data points stored in array `y[0..n-1]`.
/// The number n must be a power of 2. On exit, y is replaced by its transform. This program,
/// without changes, also calculates the inverse sine transform, but in this case the output array
/// should be multiplied by `2 / n`.
pub fn sinft(y: &mut [f64]) {
    let n = y.len();
    let mut sum;
    let mut y1;
    let mut y2;

    let mut wi = 0.0;
    let mut wr = 1.0;

    let mut wtemp;
    let theta = std::f64::consts::PI / n as f64; // Initialize the recurrence.
    wtemp = (0.5 * theta).sin();
    let wpr = -2.0 * wtemp * wtemp;
    let wpi = (theta).sin();
    y[0] = 0.0;
    for j in 1..((n >> 1) + 1) {
        wtemp = wr;
        wr += wtemp * wpr - wi * wpi; //  Calculate the sine for the auxiliary array.
        wi = wi * wpr + wtemp * wpi + wi; // The cosine is needed to continue the recurrence.
        y1 = wi * (y[j] + y[n - j]); // Construct the auxiliary array.
        y2 = 0.5 * (y[j] - y[n - j]);
        y[j] = y1 + y2; // Terms j and N - j are related.
        y[n - j] = y1 - y2;
    }
    y[0] *= 0.5;
    realft(y, 1); // Transform the auxiliary array.
    // Initialize the sum used for odd terms below.
    y[1] = 0.0;
    sum = y[1];
    for j in (0..(n - 1)).step_by(2) {
        sum += y[j];
        y[j] = y[j + 1]; // Even terms determined directly.
        y[j + 1] = sum; // Odd terms determined by this running sum.
    }
}

pub fn cosft1(y: &mut [f64]) {
    let n = y.len() - 1;
    let mut sum;
    let mut y1;
    let mut y2;

    let mut wi = 0.0;

    let mut wr = 1.0;
    let mut wtemp;
    let mut yy = vec![0.0; n]; // Need array of length n, not n+1, for realft.
    let theta = PI / n as f64; //Initialize the recurrence.
    wtemp = (0.5 * theta).sin();
    let wpr = -2.0 * wtemp * wtemp;
    let wpi = (theta).sin();
    sum = 0.5 * (y[0] - y[n]);
    yy[0] = 0.5 * (y[0] + y[n]);
    for j in 1..n / 2 {
        wtemp = wr;
        wr += wtemp * wpr - wi * wpi; //Carry out the recurrence.
        wi = wi * wpr + wtemp * wpi + wi;
        y1 = 0.5 * (y[j] + y[n - j]); //Calculate the auxiliary function.
        y2 = y[j] - y[n - j];
        yy[j] = y1 - wi * y2; //The values for j and N - j are related.
        yy[n - j] = y1 + wi * y2;
        sum += wr * y2; //Carry along this sum for later use in unfolding the transform.
    }

    yy[n / 2] = y[n / 2]; // y[n/2] unchanged.
    realft(&mut yy, 1); // Calculate the transform of the auxiliary function.
    (0..n).for_each(|j| {
        y[j] = yy[j];
    });

    y[n] = y[1];
    y[1] = sum; // sum is the value of F1 in equation (12.4.15).
    for j in (3..n).step_by(2) {
        // Unfold(j=3;j<n;j+=2) {
        sum += y[j]; // Equation (12.4.14).
        y[j] = sum;
    }
}

pub fn cosft2(y: &mut [f64], isign: i32) {
    let n = y.len();
    let mut sum;
    let mut sum1;
    let mut y1;
    let mut y2;
    let ytemp;

    let mut wi = 0.0;
    let mut wi1;

    let mut wr = 1.0;
    let mut wr1;
    let mut wtemp;
    let theta = 0.5 * PI / n as f64; // Initialize the recurrences.
    wr1 = theta.cos();
    wi1 = theta.sin();
    let wpr = -2.0 * wi1 * wi1;
    let wpi = 2.0 * theta;
    if isign == 1 {
        // Forward transform.
        for i in 0..n / 2 {
            y1 = 0.5 * (y[i] + y[n - 1 - i]); // Calculate the auxiliary function.
            y2 = wi1 * (y[i] - y[n - 1 - i]);
            y[i] = y1 + y2;
            y[n - 1 - i] = y1 - y2;
            wtemp = wr1;
            wr1 += wtemp * wpr - wi1 * wpi; // Carry out the recurrence.
            wi1 = wi1 * wpr + wtemp * wpi + wi1;
        }
        realft(y, 1); // Transform the auxiliary function.
        for i in (2..n).step_by(2) {
            // Even terms.
            wtemp = wr;
            wr += wtemp * wpr - wi * wpi;
            wi = wi * wpr + wtemp * wpi + wi;
            y1 = y[i] * wr - y[i + 1] * wi;
            y2 = y[i + 1] * wr + y[i] * wi;
            y[i] = y1;
            y[i + 1] = y2;
        }
        sum = 0.5 * y[1];
        for i in ((n - 1)..0).step_by(2) {
            sum1 = sum;
            sum += y[i];
            y[i] = sum1;
            // Initialize recurrence for odd terms with 1 / 2 RN/2.
            // Carry out recurrence for odd terms.
        }
    } else if (isign == -1) {
        ytemp = y[n - 1];
        for i in (n - 1..2).step_by(2) {
            y[i] = y[i - 2] - y[i]
        }
        y[1] = 2.0 * ytemp;
        for i in (2..n).step_by(2) {
            wtemp = wr;
            wr += wtemp * wpr - wi * wpi;
            wi = wi * wpr + wtemp * wpi + wi;
            y1 = y[i] * wr + y[i + 1] * wi;
            y2 = y[i + 1] * wr - y[i] * wi;
            y[i] = y1;
            y[i + 1] = y2;
            //Inverse transform.
            // Form diﬀerence of odd terms.
            // Calculate Rk and Ik.
        }
        realft(y, -1);
        for i in 0..(n / 2) {
            y1 = y[i] + y[n - 1 - i];
            y2 = (0.5 / wi1) * (y[i] - y[n - 1 - i]);
            y[i] = 0.5 * (y1 + y2);
            y[n - 1 - i] = 0.5 * (y1 - y2);

            wtemp = wr1;

            wr1 += wtemp * wpr - wi1 * wpi;
            wi1 = wi1 * wpr + wtemp * wpi + wi1;
            // Invert auxiliary array.
        }
    }
}

pub fn fourn(data: &mut [f64], nn: &[usize], isign: isize) {
    
    
    
    let mut i2rev;
    let mut i3rev;
    let mut ip1;
    let mut ip2;
    let mut ip3;
    let mut ifp1 = 0;
    let mut ifp2 = 0;
    let mut ibit;
    let mut k1;
    let mut k2;
    let mut n: usize = 0;
    let mut nprev;
    let mut nrem;
    let mut ntot = 1;
    let ndim = nn.len();
    let mut tempi;
    let mut tempr;
    let mut theta;
    let mut wi;
    let mut wpi;
    let mut wpr;
    let mut wr;
    let mut wtemp;
    for idim in 0..ndim {
        ntot *= nn[idim]; // Total no. of complex values.}
        if (ntot < 2 || ntot & (ntot - 1) != 0) {
            panic!("must have powers of 2 in fourn");
        }
        nprev = 1;
        for idim in ndim - 1..0 {
            // Main loop over the dimensions.
            n = nn[idim];
            nrem = ntot / (n * nprev);
            ip1 = nprev << 1;
            ip2 = ip1 * n;
            ip3 = ip2 * nrem;
            i2rev = 0;
            for i2 in (0..ip2).step_by(ip1) {
                if i2 < i2rev {
                    for i1 in (i2..i2 + ip1 - 1).step_by(2) {
                        for i3 in (i1..ip3).step_by(ip2) {
                            i3rev = i2rev + i3 - i2;

                            data.swap(i3, i3rev);
                            data.swap(i3 + 1, i3rev + 1);
                        }
                    }
                }
                ibit = ip2 >> 1;
                while (ibit >= ip1 && i2rev + 1 > ibit) {
                    i2rev -= ibit;
                    ibit >>= 1;
                }
                i2rev += ibit;
            }
            ifp1 = ip1; //Here begins the Danielson-Lanczos section of the routine.

            #[allow(clippy::while_immutable_condition)]
            while ifp1 < ip2 {
                
                ifp2 = ifp1 << 1;
                
                theta = isign as f64 * TAU / (ifp2 / ip1) as f64; //Initialize for the trigonometric recurrence.
                wtemp = (0.5 * theta).sin();
                // 
                wpr = -2.0 * wtemp * wtemp;
                wpi = (theta).sin();
                wr = 1.0;
                wi = 0.0;
                for i3 in (0..ifp1).step_by(ip1) {
                    for i1 in (i3..i3 + ip1 - 1).step_by(2) {
                        for i2 in (i1..ip3).step_by(ifp2) {
                            k1 = i2; //Danielson-Lanczos formula:
                            k2 = k1 + ifp1;
                            tempr = wr * data[k2] - wi * data[k2 + 1];
                            tempi = wr * data[k2 + 1] + wi * data[k2];
                            data[k2] = data[k1] - tempr;
                            data[k2 + 1] = data[k1 + 1] - tempi;
                            data[k1] += tempr;
                            data[k1 + 1] += tempi;
                        }
                    }
                }
                wtemp = wr;
                wr += wtemp * wpr - wi * wpi; //Trigonometric recurrence.
                wi = wi * wpr + wtemp * wpi + wi;
            }
            ifp1 = ifp2;
        }
        nprev *= n;
    }
}
