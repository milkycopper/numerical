use super::{Dim1Solver, SolveResult, DEFAULT_ITER_COUNT_LIMIT};
use crate::base_float::BaseFloat;
use crate::dim1_func::Dim1ContinuousFunc;

/// A [`Dim1NewtonSolver`] which implemented the [`Dim1Solver`] using the
/// newton's method to solve the root of a one-dimensional function.
///
/// `Dim1NewtonSolver` has the quadratic convergence, faster than the
/// [`super::BisectionSolver`] and the [`super::FixedPointSolver`]. But
/// `Dim1NewtonSolver` has chance of failing in converge to a valid root.
pub struct Dim1NewtonSolver<'a, T: BaseFloat> {
    start_point: T,
    error_tolerance: T,
    iter_count_limit: usize,
    func: &'a dyn Dim1ContinuousFunc<T>,
    func_first_derivative: &'a dyn Dim1ContinuousFunc<T>,
}

impl<'a, T: BaseFloat> Dim1NewtonSolver<'a, T> {
    /// BisectionSolver constructor
    ///
    /// # Arguments
    ///
    /// * `start_point` - The start point of search, better if near the root
    /// * `func` - The function to be solved
    /// * `func_first_derivative` - The first order derivative of solved function
    pub fn new(
        start_point: T,
        func: &'a impl Dim1ContinuousFunc<T>,
        func_first_derivative: &'a impl Dim1ContinuousFunc<T>,
    ) -> Self {
        Self {
            start_point,
            error_tolerance: T::EPSILON,
            iter_count_limit: DEFAULT_ITER_COUNT_LIMIT,
            func,
            func_first_derivative,
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

    pub fn solve(&self) -> SolveResult<T> {
        <Self as Dim1Solver<T>>::solve(self, self.func)
    }
}

impl<T: BaseFloat> Dim1Solver<T> for Dim1NewtonSolver<'_, T> {
    fn solve<F>(&self, _: &F) -> SolveResult<T>
    where
        F: Dim1ContinuousFunc<T> + ?Sized,
    {
        let mut root = self.start_point;

        for i in 0..self.iter_count_limit {
            let next = root - self.func.eval(root) / self.func_first_derivative.eval(root);
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
