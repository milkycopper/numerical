use super::{Dim1Solver, SolveResult, DEFAULT_ITER_COUNT_LIMIT};
use crate::{base_float::BaseFloat, dim1_func::Dim1ContinuousFunc};

/// A [`FixedPointSolver`] which implemented the [`Dim1Solver`] using the
/// fixed point iteration to solve the root of a one-dimensional function.
///
/// `FixedPointSolver` may not converge to a valid root,
/// which makes the [`Dim1Solver::solve`] returns `None`
///
/// `FixedPointSolver` has the linear convergence.
pub struct FixedPointSolver<T: BaseFloat> {
    start_point: T,
    error_tolerance: T,
    iter_count_limit: usize,
}

impl<T: BaseFloat> FixedPointSolver<T> {
    pub fn with_start_point(start_point: T) -> Self {
        Self {
            start_point,
            error_tolerance: T::EPSILON,
            iter_count_limit: DEFAULT_ITER_COUNT_LIMIT,
        }
    }

    pub fn with_error_tolerance(mut self, error_tolerance: T) -> Self {
        self.error_tolerance = error_tolerance;
        self
    }

    pub fn with_iter_count_limit(mut self, iter_count_limit: usize) -> Self {
        self.iter_count_limit = iter_count_limit;
        self
    }
}

impl<T: BaseFloat> Dim1Solver<T> for FixedPointSolver<T> {
    fn solve<F>(&self, func: &F) -> SolveResult<T>
    where
        F: Dim1ContinuousFunc<T> + ?Sized,
    {
        let mut root = self.start_point;

        for i in 0..self.iter_count_limit {
            let next = func.eval(root) + root;
            let diff = (next - root).abs();
            let relative_diff = diff / next.abs().max(T::ONE);
            root = next;
            if relative_diff < self.error_tolerance {
                return SolveResult {
                    solution: Some(root),
                    iter_count: i + 1,
                };
            }
        }

        SolveResult {
            solution: None,
            iter_count: self.iter_count_limit,
        }
    }
}
