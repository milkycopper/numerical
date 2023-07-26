use super::Dim1Solver;
use crate::base_float::BaseFloat;

/// A [`FixedPointSolver`] which implemented the [`Dim1Solver`] using the
/// fixed point iteration to solve the root of a one-dimensional function.
///
/// `FixedPointSolver` may not converge to a valid root,
/// which makes the [`Dim1Solver::solve`] returns `None`
///
/// `FixedPointSolver` has the linear convergence.
pub struct FixedPointSolver<T> {
    start_point: T,
    error_tolerance: T,
    max_iter_num: usize,
}

impl<T> FixedPointSolver<T> {
    /// FixedPointSolver constructor
    ///
    /// # Arguments
    ///
    /// * `start_point` - The start point of search, better if near the root
    /// * `error_tolerance` - The error tolerance determines the accuracy of the root
    /// * `max_iter_num` - The fixed point iteration may not converge to a valid root,
    ///    so the iteration need to be stoped at the max iteration number
    pub fn new(start_point: T, error_tolerance: T, max_iter_num: usize) -> Self {
        Self {
            start_point,
            error_tolerance,
            max_iter_num,
        }
    }
}

impl<T: BaseFloat> Dim1Solver<T> for FixedPointSolver<T> {
    fn solve(&self, func: &impl crate::dim1_func::Dim1Func<T>) -> Option<T> {
        let mut root = self.start_point;

        for i in 0..self.max_iter_num {
            let next = func.eval(root) + root;
            let diff = (next - root).abs();
            let relative_diff = diff / next.abs().max(T::ONE);
            root = next;
            if relative_diff < self.error_tolerance {
                log::info!("Fixed Point Iteration Num = {}", i + 1);
                return Some(root);
            }
        }

        None
    }
}
