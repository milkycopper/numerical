use crate::helpers::vec_zeros;

use super::{index2d::Index2D, Matrix, MatrixAddSubSelf, MatrixMulSelf, MatrixSquareFullVec};
use core::ops::{Deref, DerefMut, Index, IndexMut};
use core_float::CoreFloat;

/// Inner concrete type of matrix, store all matrix elements in sequence by linear Vec.
///
/// If the matrix has a shape `M * N`, then the length of inner Vec is `M * N`, and
/// elements of matrix is stored row by row.
#[derive(Clone, Eq, PartialEq)]
pub struct FullVec<T>(Vec<T>);

impl<T> Deref for FullVec<T> {
    type Target = Vec<T>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for FullVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T> Index<Index2D> for Matrix<T, FullVec<T>> {
    type Output = T;

    fn index(&self, index: Index2D) -> &Self::Output {
        &self.inner[index.full_to_1d(self.col_size())]
    }
}

impl<T> IndexMut<Index2D> for Matrix<T, FullVec<T>> {
    fn index_mut(&mut self, index: Index2D) -> &mut Self::Output {
        let index1d = index.full_to_1d(self.col_size());
        &mut self.inner[index1d]
    }
}

impl<T: CoreFloat> MatrixAddSubSelf<T> for Matrix<T, FullVec<T>> {
    fn add(&self, rhs: &Self) -> Self {
        let mut inner = vec_zeros(self.shape.area_size());
        for (i, x) in inner.iter_mut().enumerate().take(self.shape.area_size()) {
            *x = self.inner[i] + rhs.inner[i];
        }

        Self::new(self.shape, FullVec(inner))
    }

    fn add_assign(&mut self, rhs: &Self) {
        for i in 0..self.shape.area_size() {
            self.inner[i] += rhs.inner[i];
        }
    }

    fn sub(&self, rhs: &Self) -> Self {
        let mut inner = vec_zeros(self.shape.area_size());
        for (i, x) in inner.iter_mut().enumerate().take(self.shape.area_size()) {
            *x = self.inner[i] - rhs.inner[i];
        }

        Self::new(self.shape, FullVec(inner))
    }

    fn sub_assign(&mut self, rhs: &Self) {
        for i in 0..self.shape.area_size() {
            self.inner[i] -= rhs.inner[i];
        }
    }
}

impl<T: CoreFloat> MatrixMulSelf<T> for Matrix<T, FullVec<T>> {
    fn mul(&self, rhs: &Self) -> Self {
        assert!(self.col_size() == rhs.row_size());
        let shape = Index2D::from((self.row_size(), rhs.col_size()));
        let mut mat = Self::new_with_vec(shape, vec_zeros(shape.area_size()));
        for row in 0..self.row_size() {
            for col in 0..rhs.col_size() {
                for i in 0..self.col_size() {
                    mat[(row, col)] += self[(row, i)] * rhs[(i, col)];
                }
            }
        }

        mat
    }
}

/// Matrix whose inner storage type is [`FullVec`]
pub type MatrixFullVec<T> = Matrix<T, FullVec<T>>;

impl<T: CoreFloat> MatrixFullVec<T> {
    pub fn new_with_vec(shape: Index2D, v: Vec<T>) -> Self {
        debug_assert!(shape.area_size() == v.len());
        Self::new(shape, FullVec(v))
    }

    pub fn transpose(&self) -> Self {
        let mut m = Self::new_with_vec(self.shape.transpose(), vec_zeros(self.shape.area_size()));
        for j in 0..self.col_size() {
            for i in 0..self.row_size() {
                m[(j, i)] = self[(i, j)];
            }
        }

        m
    }
}

super::impl_index_usize_tuple!(MatrixFullVec<T>);

impl<T: CoreFloat> From<MatrixSquareFullVec<T>> for MatrixFullVec<T> {
    fn from(sq_m: MatrixSquareFullVec<T>) -> Self {
        MatrixFullVec::new_with_vec(sq_m.shape(), sq_m.get_inner_vec())
    }
}
