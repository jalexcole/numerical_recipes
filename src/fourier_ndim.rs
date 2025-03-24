use std::f64::consts::TAU;

use crate::fourier::fourn;

/// Given a three-dimensional real array data[0..nn1-1][0..nn2-1][0..nn3-1]
/// (where nn1 D 1 for the case of a logically two-dimensional array), this
/// routine returns (for isign=1) the complex fast Fourier transform as two
/// complex arrays: On output, data contains the zero and positive frequency
/// values of the third frequency component, while speq[0..nn1-1][0..2*nn2-1]
/// contains the Nyquist critical frequency values of the third frequency
/// component. First (and second) frequency components are stored for zero,
/// positive, and negative frequencies, in standard wraparound order. See text
/// for description of how complex values are arranged. For isign=-1, the
/// inverse transform (times nn1*nn2*nn3/2 as a constant multiplicative factor)
/// is performed, with output data (viewed as a real array) deriving from input
/// data (viewed as complex) and speq. For inverse transforms on data not
/// generated first by a forward transform, make sure the complex input data
/// array satisfies property (12.6.2). The dimensions nn1, nn2, nn3 must always
/// be integer powers of 2.
///
/// [Numerical Recipes 3rd Edition](https://numerical.recipes/book.html)
///
/// # Parameters
/// - `data`: Flattened 3D real array of dimensions [nn1][nn2][nn3].
/// - `speq`: Flattened 2D array for storing critical Nyquist frequency values,
///           with dimensions [nn1][2*nn2].
/// - `isign`:  1 for forward transform, -1 for inverse transform.
/// - `nn1`, `nn2`, `nn3`: Dimensions of the data array. (They must be powers of 2.)
pub fn rlft3(data: &mut [f64], speq: &mut [f64], isign: isize, nn1: usize, nn2: usize, nn3: usize) {
    // Check that speq has the expected size: nn1 rows, each with 2*nn2 elements.
    assert_eq!(
        speq.len(),
        nn1 * 2 * nn2,
        "speq length must equal nn1 * 2 * nn2"
    );

    let mut i2;

    let mut j1;
    let mut j2;
    let mut j3;
    let mut k1;
    let mut k2;
    let mut k3;
    let mut k4;

    let mut wi;

    let mut wr;

    let mut h1r;
    let mut h1i;
    let mut h2r;
    let mut h2i;

    let nn = [nn1, nn2, nn3];
    // Create a vector of mutable slices into speq.
    // Each slice corresponds to a row of length 2*nn2.
    let mut spq: Vec<&mut [f64]> = speq.chunks_mut(2 * nn2).collect();

    for i1 in 0..nn1 {
        // spq[i1] = speq + 2.0*nn2*i1;
    }

    let c1 = 0.5;
    let c2 = -0.5 * isign as f64;

    let theta = isign as f64 * TAU / nn3 as f64;
    let wtemp = (0.5 * theta).sin();
    let wpr = -2.0 * wtemp * wtemp;
    let wpi = theta.sin();

    if isign == 1 {
        fourn(data, &nn, isign);
        k1 = 0;
        for i1 in 0..nn1 {
            i2 = 0;
            let mut j2 = 0;
            for _ in 0..nn2 {
                spq[i1][j2] = data[k1];
                j2 += 1;
                spq[i1][j2] = data[k1 + 1];
                j2 += 1;
                k1 += nn3;
            }
        }
    }

    for i1 in 0..nn1 {
        j1 = match i1 != 0 {
            true => nn1 - i1,
            false => 0,
        };

        wr = 1.0;
        wi = 0.0;
        for i3 in (0..nn3 >> 1).step_by(2) {
            k1 = i1 * nn2 * nn3;
            k3 = j1 * nn2 * nn3;
            for i2 in 0..nn2 {
                if i3 == 0 {
                    // Equation (12.3.6).
                    j2 = match i2 != 0 {
                        true => (nn2 - i2) << 1,
                        false => 0,
                    };
                    h1r = c1 * (data[k1] + spq[j1][j2]);
                    h1i = c1 * (data[k1 + 1] - spq[j1][j2 + 1]);
                    h2i = c2 * (data[k1] - spq[j1][j2]);
                    h2r = -c2 * (data[k1 + 1] + spq[j1][j2 + 1]);
                    data[k1] = h1r + h2r;
                    data[k1 + 1] = h1i + h2i;
                    spq[j1][j2] = h1r - h2r;
                    spq[j1][j2 + 1] = h2i - h1i;
                } else {
                    j2 = match i2 != 0 {
                        true => nn2 - i2,
                        false => 0,
                    };
                    j3 = nn3 - i3;
                    k2 = k1 + i3;
                    k4 = k3 + j2 * nn3 + j3;
                    h1r = c1 * (data[k2] + data[k4]);
                    h1i = c1 * (data[k2 + 1] - data[k4 + 1]);
                    h2i = c2 * (data[k2] - data[k4]);
                    h2r = -c2 * (data[k2 + 1] + data[k4 + 1]);
                    data[k2] = h1r + wr * h2r - wi * h2i;
                    data[k2 + 1] = h1i + wr * h2i + wi * h2r;
                    data[k4] = h1r - wr * h2r + wi * h2i;
                    data[k4 + 1] = -h1i + wr * h2i + wi * h2r;

                    k1 += nn3;
                }
            }
        }
    }
}
