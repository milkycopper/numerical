use std::ops::{Deref, DerefMut};

use crate::dim1_func::Dim1ContinuousFunc;

const DEFAULT_ITER_COUNT_LIMIT: usize = 100;

pub struct SolveResult<T> {
    solution: Option<T>,
    iter_count: usize,
}

impl<T> SolveResult<T> {
    pub fn iter_count(&self) -> usize {
        self.iter_count
    }
}

impl<T> Deref for SolveResult<T> {
    type Target = Option<T>;
    fn deref(&self) -> &Self::Target {
        &self.solution
    }
}

impl<T> DerefMut for SolveResult<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.solution
    }
}

/// Solver used to finding the zero point of a continuous function
pub trait Dim1Solver<T> {
    fn solve<F>(&self, func: &F) -> SolveResult<T>
    where
        F: Dim1ContinuousFunc<T> + ?Sized;
}

mod bisection;
mod brent;
mod dim1_newton;
mod fixed_point;

pub use bisection::BisectionSolver;
pub use brent::BrentSolver;
pub use dim1_newton::Dim1NewtonSolver;
pub use fixed_point::FixedPointSolver;
