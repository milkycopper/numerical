use crate::{
    helpers::vec_zeros,
    tensor::vector::{Vector, VectorInnerVec},
};

use super::{
    base_traits::Square, index2d::Index2D, Matrix, MatrixAddSubSelf, MatrixLTVec, MatrixMulSelf,
};
use core::ops::{Deref, DerefMut, Index, IndexMut};

use core_float::CoreFloat;

/// Inner concrete type of matrix, store upper triangular elements of square matrix in sequence by linear Vec.
///
/// If the matrix has a shape `N * N`, then the length of inner Vec is `N * (N + 1) / 2`,
/// and elements of matrix is stored row by row.
#[derive(Clone, Eq, PartialEq)]
pub struct UpperTriangularVec<T> {
    s: Vec<T>,
    zero: T,
}

impl<T: CoreFloat> UpperTriangularVec<T> {
    pub fn from_vec(v: Vec<T>) -> Self {
        Self {
            s: v,
            zero: T::ZERO,
        }
    }
}

impl<T> Deref for UpperTriangularVec<T> {
    type Target = Vec<T>;
    fn deref(&self) -> &Self::Target {
        &self.s
    }
}

impl<T> DerefMut for UpperTriangularVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.s
    }
}

impl<T: CoreFloat> Index<Index2D> for Matrix<T, UpperTriangularVec<T>> {
    type Output = T;

    fn index(&self, index: Index2D) -> &Self::Output {
        if index.col < index.row {
            &self.zero
        } else {
            &self.inner[index.ut_to_1d(self.size())]
        }
    }
}

impl<T: CoreFloat> IndexMut<Index2D> for Matrix<T, UpperTriangularVec<T>> {
    fn index_mut(&mut self, index: Index2D) -> &mut Self::Output {
        debug_assert!(index.row <= index.col);
        let index1d = index.ut_to_1d(self.size());
        &mut self.inner[index1d]
    }
}

impl<T: CoreFloat> MatrixAddSubSelf<T> for Matrix<T, UpperTriangularVec<T>> {
    fn add(&self, rhs: &Self) -> Self {
        let mut inner = vec_zeros(self.elements_num());
        for (i, x) in inner.iter_mut().enumerate().take(self.elements_num()) {
            *x = self.inner[i] + rhs.inner[i];
        }

        Self::new(self.shape, UpperTriangularVec::from_vec(inner))
    }

    fn add_assign(&mut self, rhs: &Self) {
        for i in 0..self.elements_num() {
            self.inner[i] += rhs.inner[i];
        }
    }

    fn sub(&self, rhs: &Self) -> Self {
        let mut inner = vec_zeros(self.elements_num());
        for (i, x) in inner.iter_mut().enumerate().take(self.elements_num()) {
            *x = self.inner[i] - rhs.inner[i];
        }

        Self::new(self.shape, UpperTriangularVec::from_vec(inner))
    }

    fn sub_assign(&mut self, rhs: &Self) {
        for i in 0..self.elements_num() {
            self.inner[i] -= rhs.inner[i];
        }
    }
}

impl<T: CoreFloat> MatrixMulSelf<T> for Matrix<T, UpperTriangularVec<T>> {
    fn mul(&self, rhs: &Self) -> Self {
        assert!(self.size() == rhs.size());
        let mut mat = Self::new_with_vec(self.size(), vec_zeros(self.elements_num()));

        for row in 0..self.size() {
            for col in row..self.size() {
                for i in row..=col {
                    mat[(row, col)] += self[(row, i)] * rhs[(i, col)];
                }
            }
        }

        mat
    }
}

/// Matrix whose inner storage type is `UpperTriangularVec`
pub type MatrixUTVec<T> = Matrix<T, UpperTriangularVec<T>>;

impl<T: CoreFloat> MatrixUTVec<T> {
    pub fn new_with_vec(size: usize, v: Vec<T>) -> Self {
        debug_assert!(size * (size + 1) / 2 == v.len());
        Self::new((size, size).into(), UpperTriangularVec::from_vec(v))
    }

    pub fn transpose(&self) -> MatrixLTVec<T> {
        let mut m = MatrixLTVec::new_with_vec(self.size(), vec_zeros(self.elements_num()));
        for i in 0..self.size() {
            for j in i..self.size() {
                m[(j, i)] = self[(i, j)];
            }
        }

        m
    }

    /// Extend the `N * N` upper triangular matrix to `(N + 1) * (N + 1)` by inserting the `N + 1` diagonal elements
    pub fn extend_with_diagonal<I>(&self, diagonal: &mut I) -> Self
    where
        I: Iterator<Item = T>,
    {
        let size = self.size() + 1;

        let mut m = MatrixUTVec::new_with_vec(size, vec_zeros(self.elements_num() + size));

        for i in 0..size {
            m[(i, i)] = diagonal.next().unwrap();

            for j in (i + 1)..size {
                m[(i, j)] = self[(i, j - 1)];
            }
        }

        assert!(diagonal.next().is_none());

        m
    }

    /// Solving `U * x = b` for `x`, `U` is the lower triangular matrix
    pub fn back_substitution(&self, b: &VectorInnerVec<T>) -> VectorInnerVec<T> {
        assert!(self.size() == b.len());

        let mut x = Vector::new(vec_zeros(self.size()));

        let n = self.size();
        for i in (0..n).rev() {
            let mut known = T::ZERO;

            for j in (i + 1)..n {
                known += self[(i, j)] * x[j];
            }

            x[i] = (b[i] - known) / self[(i, i)];
        }

        x
    }

    #[inline]
    const fn elements_num(&self) -> usize {
        self.shape.row * (self.shape.row + 1) / 2
    }
}

impl<T> Square for MatrixUTVec<T> {
    #[inline]
    fn size(&self) -> usize {
        self.shape.row
    }
}

super::impl_index_usize_tuple!(MatrixUTVec<T>);
