use crate::{
    dim1_func::Dim1Func,
    tensor::vector::{LinearStorageLen, VecStorage, Vector},
};
use core_float::core_float_traits::CoreFloat;

/// Polynomial represented by its coefficients and base points
///
/// # Generic
/// - T - Element type
/// - S1 - Coefficients storage type
/// - S2 - Base points storage type
pub struct Polynomial<T, S1: VecStorage<T>, S2: VecStorage<T>> {
    degree: usize,
    coefficients: Vector<T, S1>,
    base_points: Option<Vector<T, S2>>,
}

impl<T, S1, S2> Polynomial<T, S1, S2>
where
    S1: VecStorage<T>,
    S2: VecStorage<T>,
    T: CoreFloat,
{
    /// Polynomial constructor
    ///
    /// # Arguments
    ///
    /// * `coefficients` - The coefficients of polynomial
    /// * `base_points` - See the **Example** section
    ///
    /// # Panic
    ///
    /// - Will panic if `coefficients` is empty
    /// - Will panic if `base_points.is_none() || base_points.as_ref().unwrap().len() != coefficients.len() - 1`
    ///
    /// # Example
    ///
    /// A polynomial `f(x) = c0 + (x − b1)(c1 + (x − b2)(c2 + (x − b3)(c3 + (x − b4)(c4))))` has
    /// coefficients array = `[c0, c1, c2, c3, c4]` and base points array = `[b1, b2, b3, b4]`
    ///
    /// ```
    /// use numerical::polynomial::Polynomial;
    /// use numerical::dim1_func::Dim1Func;
    /// use numerical::tensor::vector::Vector;
    ///
    /// let p = Polynomial::<f32, Vec<f32>, Vec<f32>>::new(vec![1., 2., 3.].into(), Some(vec![4., 5.].into()));
    /// let f = |x| 1. + (x - 4.) * (2. + (x - 5.) * 3.);
    /// for x in [7., 8., 9.] {
    ///     assert!(f(x) == p.eval(x));
    /// }
    /// ```
    pub fn new(coefficients: Vector<T, S1>, base_points: Option<Vector<T, S2>>) -> Self {
        let coefficients_len = coefficients.len();
        assert!(coefficients_len > 0);

        let degree = coefficients.len() - 1;
        assert!(base_points.is_none() || base_points.as_ref().unwrap().len() == degree);
        Self {
            degree,
            coefficients,
            base_points,
        }
    }

    /// Empty base points
    pub fn from_coefficients(coefficients: Vector<T, S1>) -> Self {
        Self::new(coefficients, None)
    }

    pub fn from_coefficients_base_points(
        coefficients: Vector<T, S1>,
        base_points: Vector<T, S2>,
    ) -> Self {
        Self::new(coefficients, Some(base_points))
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

/// Polynomial with `Vec<T>` inner storage
pub type PolynomialInnerVec<T> = Polynomial<T, Vec<T>, Vec<T>>;

impl<T, S1, S2> Dim1Func<T> for Polynomial<T, S1, S2>
where
    S1: VecStorage<T>,
    S2: VecStorage<T>,
    T: CoreFloat,
{
    fn eval(&self, x: T) -> T {
        self.nest_mul(x)
    }
}
