use super::{Index2D, Matrix, MatrixOps, MatrixShape};
use core::ops::{Deref, DerefMut, Index, IndexMut};

use core_float::CoreFloat;

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
        &self.inner[index.to_1d(self.col_size())]
    }
}

impl<T> IndexMut<Index2D> for Matrix<T, FullVec<T>> {
    fn index_mut(&mut self, index: Index2D) -> &mut Self::Output {
        let index1d = index.to_1d(self.col_size());
        &mut self.inner[index1d]
    }
}

impl<T: CoreFloat> MatrixOps<T> for Matrix<T, FullVec<T>> {
    fn default_with_shape(shape: Index2D) -> Self {
        Self::new(shape, FullVec(vec![T::ZERO; shape.col * shape.row]))
    }

    fn add(&self, rhs: &Self) -> Self {
        assert!(self.shape_eq(rhs));
        let mut mat = Self::default_with_shape(self.shape());
        for i in 0..self.inner.len() {
            mat.inner[i] = self.inner[i] + rhs.inner[i];
        }

        mat
    }

    fn add_assign(&mut self, rhs: &Self) {
        assert!(self.shape_eq(rhs));
        for i in 0..self.inner.len() {
            self.inner[i] += rhs.inner[i];
        }
    }

    fn sub(&self, rhs: &Self) -> Self {
        assert!(self.shape_eq(rhs));
        let mut mat = Self::default_with_shape(self.shape());
        for i in 0..self.inner.len() {
            mat.inner[i] = self.inner[i] - rhs.inner[i];
        }

        mat
    }

    fn sub_assign(&mut self, rhs: &Self) {
        assert!(self.shape_eq(rhs));
        for i in 0..self.inner.len() {
            self.inner[i] -= rhs.inner[i];
        }
    }
}

pub type MatrixInnerFullVec<T> = Matrix<T, FullVec<T>>;

impl<T> MatrixInnerFullVec<T> {
    pub fn new_with_vec(shape: Index2D, v: Vec<T>) -> Self {
        Self::new(shape, FullVec(v))
    }
}
