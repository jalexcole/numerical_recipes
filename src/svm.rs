


pub struct Svmgenkernel<'a>{
    /// Number of data points
    m: usize,
    /// counter for kernel calls
    kcalls: usize,
    /// Kernel matrix
    ker: Vec<Vec<f64>>,
    y: &'a [f64],
    data: &'a [&'a [f64]],
    

}