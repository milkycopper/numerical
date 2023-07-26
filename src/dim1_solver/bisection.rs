use crate::{base_float::BaseFloat, dim1_func::Dim1Func};
use core_float::core_float_traits::CoreFloat;

use super::Dim1Solver;

/// A [`BisectionSolver`] which implemented the [`Dim1Solver`] using the
/// binary search method to solve the root of a one-dimensional function.
///
/// `BisectionSolver` alway converge to a valid root if an appropriate search
/// range is provided, it has the linear convergence.
pub struct BisectionSolver<T> {
    search_range: [T; 2],
    error_tolerance: T,
}

impl<T: CoreFloat> BisectionSolver<T> {
    /// BisectionSolver constructor
    ///
    /// # Arguments
    ///
    /// * `search_range` - The root is expected be within this range `[a, b]`
    /// * `error_tolerance` - The error tolerance determines the accuracy of the root
    ///
    /// # Panic
    ///
    /// - Will panic if `search_range[0] >= search_range[1]`
    /// - Will panic if `error_tolerance <= T::EPSILON`, which means
    ///   the accuracy of the root will be limited by the machine epsilon
    pub fn new(search_range: [T; 2], error_tolerance: T) -> Self {
        assert!(search_range[0] < search_range[1]);
        assert!(error_tolerance > T::EPSILON);
        Self {
            search_range,
            error_tolerance,
        }
    }
}

impl<T: BaseFloat> Dim1Solver<T> for BisectionSolver<T> {
    fn solve(&self, func: &impl Dim1Func<T>) -> Option<T> {
        let (mut a, mut b) = (self.search_range[0], self.search_range[1]);

        assert!(func.eval(a).sig_ne(func.eval(b)));

        let mut iter_num: usize = 0;

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

            iter_num += 1;
        }

        log::info!("Bisection solve, iteration number = {}", iter_num);

        Some((b + a).half())
    }
}
