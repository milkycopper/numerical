pub mod bisection;
pub mod brent;
pub mod dim1_newton;
pub mod fixed_point;

use crate::dim1_func::Dim1Func;

/// Solver used to finding the zero point of a function
pub trait Dim1Solver<T> {
    fn solve(&self, func: &impl Dim1Func<T>) -> Option<T>;
}

pub use bisection::BisectionSolver;
pub use brent::BrentSolver;
pub use dim1_newton::Dim1NewtonSolver;
pub use fixed_point::FixedPointSolver;
