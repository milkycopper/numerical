use super::{base_traits::Square, index2d::Index2D, Matrix, MatrixBaseOps};
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

impl<T: CoreFloat> MatrixBaseOps<T> for Matrix<T, UpperTriangularVec<T>> {
    fn add(&self, rhs: &Self) -> Self {
        let mut inner = vec![T::ZERO; self.elements_num()];
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
        let mut inner = vec![T::ZERO; self.elements_num()];
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

    fn mul(&self, rhs: &Self) -> Self {
        assert!(self.size() == rhs.size());
        let mut mat = Self::new_with_vec(self.size(), vec![T::ZERO; self.elements_num()]);

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
