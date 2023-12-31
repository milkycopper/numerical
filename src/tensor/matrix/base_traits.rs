use super::{Index2D, Matrix};
use core::ops::Index;
use core_float::CoreFloat;
use std::marker::PhantomData;

macro_rules! impl_index_usize_tuple {
    ($m: ty) => {
        impl<T: CoreFloat> Index<(usize, usize)> for $m {
            type Output = T;

            fn index(&self, index: (usize, usize)) -> &Self::Output {
                &self[Index2D::from(index)]
            }
        }

        impl<T: CoreFloat> IndexMut<(usize, usize)> for $m {
            fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
                &mut self[Index2D::from(index)]
            }
        }
    };
}

pub(super) use impl_index_usize_tuple;

/// Base operations for matrix
pub trait MatrixAddSubSelf<T: CoreFloat> {
    fn add(&self, rhs: &Self) -> Self;
    fn add_assign(&mut self, rhs: &Self);
    fn sub(&self, rhs: &Self) -> Self;
    fn sub_assign(&mut self, rhs: &Self);
}

pub trait MatrixMulSelf<T: CoreFloat> {
    fn mul(&self, rhs: &Self) -> Self;
}

/// Trait for square matrix
pub trait Square {
    fn size(&self) -> usize;
}

pub struct MatrixRowByRowIter<'a, T: CoreFloat, M: Index<Index2D, Output = T>> {
    matrix: &'a M,
    phantom: PhantomData<T>,
    index2d: Index2D,
}

impl<'a, T: CoreFloat, S> Matrix<T, S>
where
    Matrix<T, S>: Index<Index2D, Output = T>,
{
    /// Return a iterator over matrix elements row by row.
    ///
    /// # Caution
    ///
    /// Use with caution for sparse matrix, because it could be very slow
    pub fn row_by_row_iter(&'a self) -> MatrixRowByRowIter<'a, T, Matrix<T, S>> {
        MatrixRowByRowIter {
            matrix: self,
            phantom: PhantomData,
            index2d: (0, 0).into(),
        }
    }
}

impl<'a, T: CoreFloat, S> Iterator for MatrixRowByRowIter<'a, T, Matrix<T, S>>
where
    Matrix<T, S>: Index<Index2D, Output = T>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let shape = self.matrix.shape;
        let current = self.index2d;

        let next_index = if current.col + 1 == shape.col {
            (current.row + 1, 0)
        } else {
            (current.row, current.col + 1)
        };

        if self.index2d.row < shape.row {
            self.index2d = next_index.into();
            Some(self.matrix[current])
        } else {
            None
        }
    }
}

mod traits_impl {
    use crate::tensor::matrix::Index2D;

    use super::super::Matrix;
    use super::{MatrixAddSubSelf, MatrixMulSelf};
    use approx::AbsDiffEq;
    use core::fmt::Display;
    use core::ops::Index;
    use core::ops::{Add, AddAssign, Mul, Sub, SubAssign};
    use core_float::core_float_traits::CoreFloat;

    impl<T, S> Add for Matrix<T, S>
    where
        T: CoreFloat,
        Matrix<T, S>: MatrixAddSubSelf<T>,
    {
        type Output = Self;
        fn add(self, rhs: Self) -> Self::Output {
            <Self as MatrixAddSubSelf<T>>::add(&self, &rhs)
        }
    }

    impl<T, S> Add<&Self> for Matrix<T, S>
    where
        T: CoreFloat,
        Matrix<T, S>: MatrixAddSubSelf<T>,
    {
        type Output = Self;
        fn add(self, rhs: &Self) -> Self::Output {
            <Self as MatrixAddSubSelf<T>>::add(&self, rhs)
        }
    }

    impl<'a, T, S> Add<Self> for &'a Matrix<T, S>
    where
        T: CoreFloat,
        Matrix<T, S>: MatrixAddSubSelf<T>,
    {
        type Output = Matrix<T, S>;
        fn add(self, rhs: Self) -> Self::Output {
            MatrixAddSubSelf::<T>::add(self, rhs)
        }
    }

    impl<T, S> AddAssign for Matrix<T, S>
    where
        T: CoreFloat,
        Matrix<T, S>: MatrixAddSubSelf<T>,
    {
        fn add_assign(&mut self, rhs: Self) {
            <Self as MatrixAddSubSelf<T>>::add_assign(self, &rhs);
        }
    }

    impl<T, S> AddAssign<&Self> for Matrix<T, S>
    where
        T: CoreFloat,
        Matrix<T, S>: MatrixAddSubSelf<T>,
    {
        fn add_assign(&mut self, rhs: &Self) {
            <Self as MatrixAddSubSelf<T>>::add_assign(self, rhs);
        }
    }

    impl<T, S> Sub for Matrix<T, S>
    where
        T: CoreFloat,
        Matrix<T, S>: MatrixAddSubSelf<T>,
    {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self::Output {
            <Self as MatrixAddSubSelf<T>>::sub(&self, &rhs)
        }
    }

    impl<T, S> Sub<&Self> for Matrix<T, S>
    where
        T: CoreFloat,
        Matrix<T, S>: MatrixAddSubSelf<T>,
    {
        type Output = Self;
        fn sub(self, rhs: &Self) -> Self::Output {
            <Self as MatrixAddSubSelf<T>>::sub(&self, rhs)
        }
    }

    impl<'a, T, S> Sub<Self> for &'a Matrix<T, S>
    where
        T: CoreFloat,
        Matrix<T, S>: MatrixAddSubSelf<T>,
    {
        type Output = Matrix<T, S>;
        fn sub(self, rhs: Self) -> Self::Output {
            MatrixAddSubSelf::<T>::sub(self, rhs)
        }
    }

    impl<T, S> SubAssign for Matrix<T, S>
    where
        T: CoreFloat,
        Matrix<T, S>: MatrixAddSubSelf<T>,
    {
        fn sub_assign(&mut self, rhs: Self) {
            <Self as MatrixAddSubSelf<T>>::sub_assign(self, &rhs);
        }
    }

    impl<T, S> SubAssign<&Self> for Matrix<T, S>
    where
        T: CoreFloat,
        Matrix<T, S>: MatrixAddSubSelf<T>,
    {
        fn sub_assign(&mut self, rhs: &Self) {
            <Self as MatrixAddSubSelf<T>>::sub_assign(self, rhs);
        }
    }

    impl<T, S> Mul<&Self> for Matrix<T, S>
    where
        T: CoreFloat,
        Matrix<T, S>: MatrixMulSelf<T>,
    {
        type Output = Self;
        fn mul(self, rhs: &Self) -> Self::Output {
            <Self as MatrixMulSelf<T>>::mul(&self, rhs)
        }
    }

    impl<'a, T, S> Mul<Self> for &'a Matrix<T, S>
    where
        T: CoreFloat,
        Matrix<T, S>: MatrixMulSelf<T>,
    {
        type Output = Matrix<T, S>;
        fn mul(self, rhs: Self) -> Self::Output {
            MatrixMulSelf::<T>::mul(self, rhs)
        }
    }

    impl<T, S> Display for Matrix<T, S>
    where
        T: CoreFloat,
        Matrix<T, S>: Index<Index2D, Output = T>,
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            writeln!(f, "[")?;
            for i in 0..self.row_size() {
                write!(f, "[")?;
                for j in 0..self.col_size() {
                    write!(f, "{}", self[(i, j).into()])?;
                    if j != self.col_size() - 1 {
                        write!(f, ", ")?;
                    }
                }
                write!(f, "]")?;
                if i != self.row_size() - 1 {
                    writeln!(f, ",")?;
                }
            }
            write!(f, "\n]")?;

            Ok(())
        }
    }

    // TODO: Special implementation for matrices with different internal types
    impl<T, S> AbsDiffEq for Matrix<T, S>
    where
        S: core::cmp::PartialEq,
        T: CoreFloat + AbsDiffEq<Epsilon = T>,
        Matrix<T, S>: Index<Index2D, Output = T>,
    {
        type Epsilon = T;
        fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
            for (a, b) in self.row_by_row_iter().zip(other.row_by_row_iter()) {
                if !a.abs_diff_eq(&b, epsilon) {
                    return false;
                }
            }

            true
        }

        fn default_epsilon() -> Self::Epsilon {
            T::EPSILON
        }
    }
}
