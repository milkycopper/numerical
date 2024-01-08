use std::{marker::PhantomData, ops::Deref};

use crate::base_float::BaseFloat;

pub trait ContinuousFunc {}

pub trait Dim1Func<T> {
    /// Evaluate `f(x)` at `x`
    fn eval(&self, x: T) -> T;
}

pub trait Dim1ContinuousFunc<T>: Dim1Func<T> + ContinuousFunc {}

impl<T, F> Dim1Func<T> for F
where
    F: Fn(T) -> T,
{
    fn eval(&self, x: T) -> T {
        self(x)
    }
}

/// It is the user's responsibility to ensure the continuity of the function
pub struct SimpleDim1ContinuousFunc<'a, T: BaseFloat> {
    func: &'a dyn Fn(T) -> T,
    phantom: PhantomData<T>,
}

impl<'a, T: BaseFloat> SimpleDim1ContinuousFunc<'a, T> {
    pub fn new(func: &'a dyn Fn(T) -> T) -> Self {
        Self {
            func,
            phantom: PhantomData::<T>,
        }
    }
}

impl<'a, T: BaseFloat> Deref for SimpleDim1ContinuousFunc<'a, T> {
    type Target = &'a dyn Fn(T) -> T;
    fn deref(&self) -> &Self::Target {
        &self.func
    }
}

impl<'a, T: BaseFloat> From<&'a dyn Fn(T) -> T> for SimpleDim1ContinuousFunc<'a, T> {
    fn from(value: &'a dyn Fn(T) -> T) -> Self {
        Self::new(value)
    }
}

impl<'a, T: BaseFloat> Dim1Func<T> for SimpleDim1ContinuousFunc<'a, T> {
    fn eval(&self, x: T) -> T {
        self.func.eval(x)
    }
}

impl<'a, T: BaseFloat> ContinuousFunc for SimpleDim1ContinuousFunc<'a, T> {}
impl<'a, T: BaseFloat> Dim1ContinuousFunc<T> for SimpleDim1ContinuousFunc<'a, T> {}
