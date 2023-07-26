use super::Dim1Solver;
use crate::base_float::BaseFloat;
use crate::dim1_func::Dim1Func;

/// A [`Dim1NewtonSolver`] which implemented the [`Dim1Solver`] using the
/// newton's method to solve the root of a one-dimensional function.
///
/// `Dim1NewtonSolver` has the quadratic convergence, faster than the
/// [`super::BisectionSolver`] and the [`super::FixedPointSolver`]. But
/// `Dim1NewtonSolver` has chance of failing in converge to a valid root.
pub struct Dim1NewtonSolver<'a, T> {
    start_point: T,
    error_tolerance: T,
    max_iter_num: usize,
    func: &'a dyn crate::dim1_func::Dim1Func<T>,
    func_first_derivative: &'a dyn crate::dim1_func::Dim1Func<T>,
}

impl<'a, T> Dim1NewtonSolver<'a, T> {
    /// BisectionSolver constructor
    ///
    /// # Arguments
    ///
    /// * `start_point` - The start point of search, better if near the root
    /// * `error_tolerance` - The error tolerance determines the accuracy of the root
    /// * `max_iter_num` - The newton's method may not converge to a valid root,
    ///    so the iteration need to be stoped at the max iteration number
    /// * `func` - The function to be solved
    /// * `func_first_derivative` - The first order derivative of solved function
    pub fn new(
        start_point: T,
        error_tolerance: T,
        max_iter_num: usize,
        func: &'a impl Dim1Func<T>,
        func_first_derivative: &'a impl Dim1Func<T>,
    ) -> Self {
        Self {
            start_point,
            error_tolerance,
            max_iter_num,
            func,
            func_first_derivative,
        }
    }
}

impl<T: BaseFloat> Dim1Solver<T> for Dim1NewtonSolver<'_, T> {
    fn solve(&self, _: &impl crate::dim1_func::Dim1Func<T>) -> Option<T> {
        let mut root = self.start_point;

        for i in 0..self.max_iter_num {
            let next = root - self.func.eval(root) / self.func_first_derivative.eval(root);
            let diff = (next - root).abs();
            let relative_diff = diff / next.abs().max(T::ONE);
            root = next;

            if relative_diff < self.error_tolerance {
                log::info!("Dim1 Newton Iteration Num = {}", i + 1);
                return Some(root);
            }
        }

        None
    }
}
