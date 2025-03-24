/// Given a function or functor func and an initial guessed range x1 to x2, the routine expands
/// the range geometrically until a root is bracketed by the returned values x1 and x2 (in which
/// case zbrac returns true) or until the range becomes unacceptably large (in which case zbrac
/// returns false).
pub fn zbrac<T>(func: &T, x1: &mut f64, x2: &mut f64) -> bool
where
    T: Fn(f64) -> f64,
{
    const NTRY: usize = 50;
    const FACTOR: f64 = 1.6;
    if (x1 == x2) {
        panic!("Bad initial range in zbrac");
    }
    let mut f1 = func(*x1);
    let mut f2 = func(*x2);

    for _ in 0..NTRY {
        if f1 * f2 < 0.0 {
            return true;
        }
        if f1.abs() < (f2).abs() {
            *x1 = (x1.clone() + FACTOR * (x1.clone() - x2.clone()));
            f1 = func(*x1);
        } else {
            *x2 = x2.clone() + FACTOR * (x2.clone() - x1.clone());
            f2 = func(*x2);
        }
    }
    return false;
}
