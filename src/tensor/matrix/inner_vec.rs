use super::{Index2D, Matrix, MatrixOps, MatrixShape};
use core::ops::{Index, IndexMut};
use core_float::CoreFloat;

impl<T> Index<Index2D> for Matrix<T, Vec<T>> {
    type Output = T;

    fn index(&self, index: Index2D) -> &Self::Output {
        &self.inner[index.to_1d(self.col_size())]
    }
}

impl<T> IndexMut<Index2D> for Matrix<T, Vec<T>> {
    fn index_mut(&mut self, index: Index2D) -> &mut Self::Output {
        let index1d = index.to_1d(self.col_size());
        &mut self.inner[index1d]
    }
}

impl<T: CoreFloat> MatrixOps<T> for Matrix<T, Vec<T>> {
    fn default_with_shape(shape: Index2D) -> Self {
        Self::new(shape, vec![T::ZERO; shape.col * shape.row])
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

pub type MatrixInnerVec<T> = Matrix<T, Vec<T>>;
