use std::sync::LazyLock;

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct State {
    lam1: f64,
    lam2: f64,
    tc: f64,
    k1: i32,
    k2: i32,
    plog: f64,
}

impl State {
    pub fn new(lam1: f64, lam2: f64, t: f64, k1: i32, k2: i32) -> Self {
        State {
            lam1,
            lam2,
            tc: t,
            k1,
            k2,
            plog: 0.0,
        }
    }

    pub fn plog(&self) -> f64 {
        self.plog
    }
}

pub struct Plog<'a> {
    dat: &'a [f64],
    ndat: usize,
    stau: Vec<f64>,
    slogtau: Vec<f64>,
}

impl<'a> Plog<'a> {
    pub fn plog(data: &'a [f64]) -> impl Fn(&mut State) -> f64 + 'a {
        let mut plog = Plog {
            dat: data,
            ndat: data.len(),
            stau: Vec::new(),
            slogtau: Vec::new(),
        };

        move |s: &mut State| -> f64 {
            let mut i;
            // let ilet;
            let mut ilo;
            let mut ihi;

            let mut ans;

            ilo = 0;
            ihi = plog.ndat - 1;
            while ihi - ilo > 1 {
                i = (ihi + ilo) >> 1;
                if s.tc > plog.dat[i] {
                    ilo = i;
                } else {
                    ihi = i;
                }
                // Bisection to find where is tc in the data.
            }
            let n1 = ihi;
            let n2 = plog.ndat - 1 - ihi;
            let st1 = plog.stau[ihi];
            let st2 = plog.stau[plog.ndat - 1] - st1;
            let stl1 = plog.slogtau[ihi];
            let stl2 = plog.slogtau[plog.ndat - 1] - stl1;
            // Equations (15.8.11) and (15.8.12):
            ans = n1 as f64 * (s.k1 as f64 * (s.lam1.ln()) 
                - factln(s.k1 as usize - 1))
                + (s.k1 - 1) as f64 * stl1
                - s.lam1 * st1;
            ans += n2 as f64 * (s.k2 as f64 * (s.lam2).log(10.0) 
                - factln(s.k2 as usize - 1))
                + (s.k2 - 1) as f64 * stl2
                - s.lam2 * st2;
            s.plog = ans;

            ans
        }
    }
}

fn factln(n: usize) -> f64 {
    // Precomputed values for small n
    static FACTLN: LazyLock<[f64; 101]> = LazyLock::new(|| {
        let mut arr = [0.0; 101];
        arr[0] = 0.0;
        for i in 1..=100 {
            arr[i] = arr[i - 1] + (i as f64).ln();
        }
        arr
    });

    if n <= 100 {
        FACTLN[n]
    } else {
        let x = n as f64;
        x * x.ln() - x + 0.5 * (2.0 * std::f64::consts::PI * x).ln()
    }
}

pub struct Proposal {
    
}
