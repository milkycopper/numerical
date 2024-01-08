use crate::{
    dim1_func::{ContinuousFunc, Dim1ContinuousFunc, Dim1Func},
    tensor::vector::VectorInnerVec as Vector,
};
use core_float::core_float_traits::CoreFloat;

/// Polynomial represented by its coefficients and base points
///
/// # Generic
/// - T - Element type which implemented [`CoreFloat`] trait
///
/// # Examples
///
/// A polynomial `f(x) = c0 + (x − b1)(c1 + (x − b2)(c2 + (x − b3)(c3 + (x − b4)(c4))))` has
/// coefficients array = `[c0, c1, c2, c3, c4]` and base points array = `[b1, b2, b3, b4]`
///
/// ```
/// use numerical::polynomial::Polynomial;
/// use numerical::dim1_func::Dim1Func;
/// use numerical::tensor::vector::Vector;
///
/// let p = Polynomial::with_coefficients(vec![1., 2., 3.].into())
///         .with_base_points(vec![4., 5.].into());
/// let f = |x| 1. + (x - 4.) * (2. + (x - 5.) * 3.);
/// for x in [7., 8., 9.] {
///     assert!(f(x) == p.eval(x));
/// }
/// ```
pub struct Polynomial<T: CoreFloat> {
    degree: usize,
    coefficients: Vector<T>,
    base_points: Option<Vector<T>>,
}

impl<T: CoreFloat> Polynomial<T> {
    pub fn with_coefficients(coefficients: Vector<T>) -> Self {
        assert!(!coefficients.is_empty());

        Self {
            degree: coefficients.len() - 1,
            coefficients,
            base_points: None,
        }
    }

    pub fn with_base_points(mut self, base_points: Vector<T>) -> Self {
        assert!(base_points.len() == self.degree);
        self.base_points = Some(base_points);
        self
    }

    /// Nested multiplication - An effective algorithm for evaluating the value of polynomial at `x`
    pub fn nest_mul(&self, x: T) -> T {
        let d = self.degree;
        let mut y: T = self.coefficients[d];

        if let Some(b) = &self.base_points {
            for i in (0..d).rev() {
                y = y * (x - b[i]) + self.coefficients[i];
            }
        } else {
            for i in (0..d).rev() {
                y = y * x + self.coefficients[i];
            }
        }

        y
    }
}

impl<T: CoreFloat> Dim1Func<T> for Polynomial<T> {
    fn eval(&self, x: T) -> T {
        self.nest_mul(x)
    }
}

impl ContinuousFunc for Polynomial<f32> {}
impl ContinuousFunc for Polynomial<f64> {}
impl Dim1ContinuousFunc<f32> for Polynomial<f32> {}
impl Dim1ContinuousFunc<f64> for Polynomial<f64> {}
