use super::{index2d::Index2D, Matrix, MatrixBaseOps};
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

impl<T: CoreFloat> MatrixBaseOps<T> for Matrix<T, FullVec<T>> {
    fn add(&self, rhs: &Self) -> Self {
        let mut inner = vec![T::ZERO; self.shape.area_size()];
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
        let mut inner = vec![T::ZERO; self.shape.area_size()];
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

    fn mul(&self, rhs: &Self) -> Self {
        assert!(self.col_size() == rhs.row_size());
        let shape = Index2D::from((self.row_size(), rhs.col_size()));
        let mut mat = Self::new_with_vec(shape, vec![T::ZERO; shape.area_size()]);
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

/// Matrix whose inner storage type is `FullVec`
pub type MatrixFullInnerVec<T> = Matrix<T, FullVec<T>>;

impl<T> MatrixFullInnerVec<T> {
    pub fn new_with_vec(shape: Index2D, v: Vec<T>) -> Self {
        debug_assert!(shape.area_size() == v.len());
        Self::new(shape, FullVec(v))
    }
}

super::impl_index_usize_tuple!(MatrixFullInnerVec<T>);
