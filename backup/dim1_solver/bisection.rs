use super::{Dim1Solver, SolveResult};
use crate::{base_float::BaseFloat, dim1_func::Dim1ContinuousFunc};

/// A [`BisectionSolver`] which implemented the [`Dim1Solver`] using the
/// binary search method to solve the root of a one-dimensional function.
///
/// `BisectionSolver` alway converge to a valid root if an appropriate search
/// range is provided, it has the linear convergence.
pub struct BisectionSolver<T: BaseFloat> {
    search_range: [T; 2],
    error_tolerance: T,
}

impl<T: BaseFloat> BisectionSolver<T> {
    pub fn with_search_range(range: [T; 2]) -> Self {
        assert!(range[0] < range[1]);
        Self {
            search_range: range,
            error_tolerance: T::EPSILON,
        }
    }

    pub fn with_error_tolerance(mut self, tolerance: T) -> Self {
        self.error_tolerance = tolerance;
        self
    }
}

impl<T: BaseFloat> Dim1Solver<T> for BisectionSolver<T> {
    fn solve<F>(&self, func: &F) -> SolveResult<T>
    where
        F: Dim1ContinuousFunc<T> + ?Sized,
    {
        let (mut a, mut b) = (self.search_range[0], self.search_range[1]);

        assert!(func.eval(a).sig_ne(func.eval(b)));

        let mut iter_count = 0;

        while (b - a) > self.error_tolerance.double() {
            let c = (b + a).half();
            let fc = func.eval(c);

            if fc == T::ZERO {
                break;
            }

            if func.eval(a).sig_ne(fc) {
                b = c;
            } else {
                debug_assert!(func.eval(b).sig_ne(fc));
                a = c;
            }

            iter_count += 1;
        }

        SolveResult {
            solution: Some((a + b).half()),
            iter_count,
        }
    }
}
