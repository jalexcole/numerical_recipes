use std::ops::{Index, IndexMut};
use std::ptr;

use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

/// A custom `Num` trait to restrict types to basic numeric operations
pub trait Num:
    Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Rem<Output = Self>
    + PartialOrd
    + Copy
    + Default
    + 'static
{
    /// Method to return the additive identity (zero)
    fn zero() -> Self;

    /// Method to return the multiplicative identity (one)
    fn one() -> Self;
}

// Implement `Num` for primitive integer and floating-point types
macro_rules! impl_num_for_primitive {
    ($($t:ty)*) => ($(
        impl Num for $t {
            #[inline]
            fn zero() -> Self { 0 as $t }

            #[inline]
            fn one() -> Self { 1 as $t }
        }
    )*)
}

impl_num_for_primitive!(u8 u16 u32 u64 u128 i8 i16 i32 i64 i128 f32 f64);

#[derive(Debug, Clone)]
pub struct NRvector<T: Num> {
    v: Vec<T>,
}

impl<T: Num> NRvector<T> {
    // Default constructor
    pub fn new() -> Self {
        NRvector { v: Vec::new() }
    }

    // Construct vector of size n
    pub fn with_size(n: usize) -> Self
    where
        T: Default + Clone,
    {
        NRvector {
            v: vec![T::default(); n],
        }
    }

    // Initialize to constant value a
    pub fn with_value(n: usize, a: T) -> Self
    where
        T: Clone,
    {
        NRvector { v: vec![a; n] }
    }

    // Initialize to values in a slice (C-style array equivalent)
    pub fn from_slice(a: &[T]) -> Self
    where
        T: Clone,
    {
        NRvector { v: a.to_vec() }
    }

    // Copy constructor
    pub fn copy_from(rhs: &NRvector<T>) -> Self
    where
        T: Clone,
    {
        NRvector { v: rhs.v.clone() }
    }

    // Assignment operator
    pub fn assign(&mut self, rhs: &NRvector<T>)
    where
        T: Clone,
    {
        self.v = rhs.v.clone();
    }

    // Return size of vector
    pub fn size(&self) -> usize {
        self.v.len()
    }

    // Resize, losing contents
    pub fn resize(&mut self, newn: usize)
    where
        T: Default + Clone,
    {
        self.v.resize(newn, T::default());
    }

    // Resize and assign `a` to every element
    pub fn assign_with_value(&mut self, newn: usize, a: T)
    where
        T: Clone,
    {
        self.v = vec![a; newn];
    }
}

// Indexing
impl<T: Num> Index<usize> for NRvector<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.v[index]
    }
}

impl<T: Num> IndexMut<usize> for NRvector<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.v[index]
    }
}

#[derive(Debug, Clone)]
pub struct NRmatrix<T: Num> {
    data: Vec<T>,
    shape: [usize; 2], // shape[0] = rows, shape[1] = columns
}

impl<T: Num> NRmatrix<T> {
    /// Default constructor: creates an empty matrix
    pub fn new() -> Self {
        NRmatrix {
            data: Vec::new(),
            shape: [0, 0],
        }
    }

    /// Construct an n x m matrix with default values
    pub fn with_size(rows: usize, cols: usize) -> Self
    where
        T: Default + Clone,
    {
        NRmatrix {
            data: vec![T::default(); rows * cols],
            shape: [rows, cols],
        }
    }

    /// Initialize to constant value `a`
    pub fn with_value(rows: usize, cols: usize, value: T) -> Self
    where
        T: Clone,
    {
        NRmatrix {
            data: vec![value; rows * cols],
            shape: [rows, cols],
        }
    }

    /// Initialize to values in a flattened slice (C-style array equivalent)
    pub fn from_slice(rows: usize, cols: usize, slice: &[T]) -> Self
    where
        T: Clone,
    {
        assert_eq!(
            slice.len(),
            rows * cols,
            "Slice length does not match matrix dimensions"
        );
        NRmatrix {
            data: slice.to_vec(),
            shape: [rows, cols],
        }
    }

    /// Copy constructor
    pub fn copy_from(rhs: &NRmatrix<T>) -> Self
    where
        T: Clone,
    {
        NRmatrix {
            data: rhs.data.clone(),
            shape: rhs.shape,
        }
    }

    /// Assignment operator
    pub fn assign(&mut self, rhs: &NRmatrix<T>)
    where
        T: Clone,
    {
        self.data = rhs.data.clone();
        self.shape = rhs.shape;
    }

    /// Return the number of rows
    pub fn nrows(&self) -> usize {
        self.shape[0]
    }

    /// Return the number of columns
    pub fn ncols(&self) -> usize {
        self.shape[1]
    }

    /// Resize, losing contents
    pub fn resize(&mut self, new_rows: usize, new_cols: usize)
    where
        T: Default + Clone,
    {
        self.data.resize(new_rows * new_cols, T::default());
        self.shape = [new_rows, new_cols];
    }

    /// Resize and assign `value` to every element
    pub fn assign_with_value(&mut self, new_rows: usize, new_cols: usize, value: T)
    where
        T: Clone,
    {
        self.data = vec![value; new_rows * new_cols];
        self.shape = [new_rows, new_cols];
    }

    /// Access an element by (row, column)
    pub fn get(&self, row: usize, col: usize) -> &T {
        assert!(
            row < self.shape[0] && col < self.shape[1],
            "Index out of bounds"
        );
        &self.data[row * self.shape[1] + col]
    }

    /// Mutable access to an element by (row, column)
    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut T {
        assert!(
            row < self.shape[0] && col < self.shape[1],
            "Index out of bounds"
        );
        &mut self.data[row * self.shape[1] + col]
    }

    pub fn swap_rows(&mut self, row1: usize, row2: usize) {
        let cols = self.ncols();
        for col in 0..cols {
            self.data.swap(row1 * cols + col, row2 * cols + col);
        }
    }

    pub fn swap_cols(&mut self, col1: usize, col2: usize) {
        let rows = self.nrows();
        let cols = self.ncols();
        for row in 0..rows {
            self.data.swap(row * cols + col1, row * cols + col2);
        }
    }
}

// Indexing for row access
impl<T: Num> Index<usize> for NRmatrix<T> {
    type Output = [T];

    fn index(&self, row: usize) -> &Self::Output {
        let start = row * self.shape[1];
        let end = start + self.shape[1];
        &self.data[start..end]
    }
}

impl<T: Num> IndexMut<usize> for NRmatrix<T> {
    fn index_mut(&mut self, row: usize) -> &mut Self::Output {
        let start = row * self.shape[1];
        let end = start + self.shape[1];
        &mut self.data[start..end]
    }
}

// Define a global switch to toggle between simple and advanced error handling
const USE_NR_ERROR_CLASS: bool = true;

/// Simple error handling (similar to the default `#define throw` macro in C++)
macro_rules! simple_throw {
    ($message:expr) => {{
        eprintln!(
            "ERROR: {}\n in file {} at line {}",
            $message,
            file!(),
            line!()
        );
        std::process::exit(1);
    }};
}

/// Advanced error class `NRerror` equivalent
/// ```rust
/// // Example function that uses `throw!` for error handling
/// fn cholesky_example(success: bool) -> Result<(), NRerror> {
///    if !success {
///        throw!("Cholesky error occurred");
///    }
///    println!("Cholesky computation succeeded.");
///    Ok(())
///}
///
/// ```
#[derive(Debug)]
pub struct NRerror {
    pub message: String,
    pub file: &'static str,
    pub line: u32,
}

impl NRerror {
    pub fn new(message: &str, file: &'static str, line: u32) -> Self {
        NRerror {
            message: message.to_string(),
            file,
            line,
        }
    }
}

impl std::fmt::Display for NRerror {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ERROR: {}\n in file {} at line {}",
            self.message, self.file, self.line
        )
    }
}

/// Function to handle the NRerror (equivalent to `NRcatch` in C++)
pub fn nrcatch(err: NRerror) {
    eprintln!("{}", err);
    std::process::exit(1);
}

/// Advanced error handling macro
macro_rules! advanced_throw {
    ($message:expr) => {
        Err(NRerror::new($message, file!(), line!()))
    };
}

/// Unified `throw!` macro that chooses between simple and advanced error handling
macro_rules! throw {
    ($message:expr) => {
        if USE_NR_ERROR_CLASS {
            return advanced_throw!($message);
        } else {
            simple_throw!($message);
        }
    };
}
