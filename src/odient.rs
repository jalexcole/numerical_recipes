use crate::stepper::Stepper;

pub struct Odeint<'a, T> {
    eps: f64,
    nok: i32,
    nbad: i32,
    nvar: i32,
    x1: f64,
    x2: f64,
    hmin: f64,
    dense: bool,
    y: Vec<f64>,
    dydx: f64,
    ystart: &'a Vec<f64>,
    s: T,
}

impl<'a, T> Odeint<'a, T> {
    const MAXSTP: usize = 50000;
}


#[derive(Debug, Clone)]
pub struct Output {
    pub kmax: isize,
    pub nvar: usize,
    pub nsave: usize,
    pub dense: bool,
    pub count: usize,
    pub x1: f64,
    pub x2: f64,
    pub xout: f64,
    pub dxout: f64,
    pub xsave: Vec<f64>,
    pub ysave: Vec<Vec<f64>>,
}

impl Output {
    /// Default constructor gives no output.
    pub fn new() -> Self {
        Output {
            kmax: -1,
            nvar: 0,
            nsave: 0,
            dense: false,
            count: 0,
            x1: 0.0,
            x2: 0.0,
            xout: 0.0,
            dxout: 0.0,
            xsave: Vec::new(),
            ysave: Vec::new(),
        }
    }

    /// Constructor provides dense output at `nsave` equally spaced intervals.
    /// If `nsave == 0`, output is saved only at the actual integration steps.
    pub fn with_nsave(nsave: usize) -> Self {
        let kmax = 500;
        Output {
            kmax,
            nvar: 0,
            nsave,
            dense: nsave > 0,
            count: 0,
            x1: 0.0,
            x2: 0.0,
            xout: 0.0,
            dxout: 0.0,
            xsave: vec![0.0; kmax as usize],
            ysave: Vec::new(),
        }
    }

    /// Called by Odeint constructor with number of equations and integration range
    pub fn init(&mut self, neqn: usize, xlo: f64, xhi: f64) {
        self.nvar = neqn;
        if self.kmax == -1 {
            return;
        }

        self.ysave = vec![vec![0.0; self.kmax as usize]; self.nvar];

        if self.dense {
            self.x1 = xlo;
            self.x2 = xhi;
            self.xout = self.x1;
            self.dxout = (self.x2 - self.x1) / self.nsave as f64;
        }
    }

    /// Resize storage arrays by a factor of two, keeping saved data
    pub fn resize(&mut self) {
        let kold = self.kmax as usize;
        self.kmax *= 2;

        // Save old data
        let tempvec = self.xsave.clone();
        let tempmat = self.ysave.clone();

        // Resize xsave and restore values
        self.xsave.resize(self.kmax as usize, 0.0);
        for k in 0..kold {
            self.xsave[k] = tempvec[k];
        }

        // Resize ysave and restore values
        for row in &mut self.ysave {
            row.resize(self.kmax as usize, 0.0);
        }

        for i in 0..self.nvar {
            for k in 0..kold {
                self.ysave[i][k] = tempmat[i][k];
            }
        }
    }
    /// Save dense output at `xout` using the stepper's dense_out method
    pub fn save_dense<S: Stepper>(&mut self, s: &S, xout: f64, h: f64) {
        if self.count == self.kmax as usize {
            self.resize();
        }

        for i in 0..self.nvar {
            // self.ysave[i][self.count] = s.dense_out(i, xout, h);
        }

        self.xsave[self.count] = xout;
        self.count += 1;
    }

    /// Save raw values of x and y
    pub fn save(&mut self, x: f64, y: &[f64]) {
        if self.kmax <= 0 {
            return;
        }

        if self.count == self.kmax as usize {
            self.resize();
        }

        for i in 0..self.nvar {
            self.ysave[i][self.count] = y[i];
        }

        self.xsave[self.count] = x;
        self.count += 1;
    }

    /// Called during integration to handle dense output
    pub fn out<S: Stepper>(
        &mut self,
        nstp: isize,
        x: f64,
        y: &[f64],
        s: &S,
        h: f64,
    ) {
        if !self.dense {
            panic!("dense output not set in Output!");
        }

        if nstp == -1 {
            self.save(x, y);
            self.xout += self.dxout;
        } else {
            while (x - self.xout) * (self.x2 - self.x1) > 0.0 {
                self.save_dense(s, self.xout, h);
                self.xout += self.dxout;
            }
        }
    }
}