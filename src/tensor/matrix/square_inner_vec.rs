use crate::{
    helpers::vec_zeros,
    tensor::{matrix::LUFactorization, vector::VectorInnerVec},
};

use super::{
    index2d::Index2D, Matrix, MatrixAddSubSelf, MatrixLTVec, MatrixMulSelf, MatrixUTVec, Square,
};
use core::ops::{Deref, DerefMut, Index, IndexMut};
use core_float::CoreFloat;

/// Inner concrete type of matrix, store all square matrix elements in sequence by linear Vec.
///
/// If the matrix has a shape `N * N`, then the length of inner Vec is `N * N`, and
/// elements of matrix is stored row by row.
#[derive(Clone, Eq, PartialEq)]
pub struct SquareFullVec<T>(Vec<T>);

impl<T> Deref for SquareFullVec<T> {
    type Target = Vec<T>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for SquareFullVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T> Index<Index2D> for Matrix<T, SquareFullVec<T>> {
    type Output = T;

    fn index(&self, index: Index2D) -> &Self::Output {
        &self.inner[index.full_to_1d(self.size())]
    }
}

impl<T> IndexMut<Index2D> for Matrix<T, SquareFullVec<T>> {
    fn index_mut(&mut self, index: Index2D) -> &mut Self::Output {
        let index1d = index.full_to_1d(self.size());
        &mut self.inner[index1d]
    }
}

impl<T: CoreFloat> MatrixAddSubSelf<T> for Matrix<T, SquareFullVec<T>> {
    fn add(&self, rhs: &Self) -> Self {
        let mut inner = vec_zeros(self.shape.area_size());
        for (i, x) in inner.iter_mut().enumerate().take(self.shape.area_size()) {
            *x = self.inner[i] + rhs.inner[i];
        }

        Self::new(self.shape, SquareFullVec(inner))
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

        Self::new(self.shape, SquareFullVec(inner))
    }

    fn sub_assign(&mut self, rhs: &Self) {
        for i in 0..self.shape.area_size() {
            self.inner[i] -= rhs.inner[i];
        }
    }
}

impl<T: CoreFloat> MatrixMulSelf<T> for Matrix<T, SquareFullVec<T>> {
    fn mul(&self, rhs: &Self) -> Self {
        assert!(self.size() == rhs.size());
        let mut mat = Self::new_with_vec(self.size(), vec_zeros(self.size() * self.size()));
        for row in 0..self.size() {
            for col in 0..rhs.size() {
                for i in 0..self.size() {
                    mat[(row, col)] += self[(row, i)] * rhs[(i, col)];
                }
            }
        }

        mat
    }
}

/// Matrix whose inner storage type is `SquareFullVec`
pub type MatrixSquareFullVec<T> = Matrix<T, SquareFullVec<T>>;

impl<T: CoreFloat> MatrixSquareFullVec<T> {
    pub fn new_with_vec(size: usize, v: Vec<T>) -> Self {
        debug_assert!(size * size == v.len());
        Self::new((size, size).into(), SquareFullVec(v))
    }

    pub fn transpose(&self) -> Self {
        let mut m = Self::new_with_vec(self.size(), vec_zeros(self.shape.area_size()));
        for j in 0..self.size() {
            for i in 0..self.size() {
                m[(j, i)] = self[(i, j)];
            }
        }

        m
    }

    /// Matrix multiply a vector, performing a linear transformation on the vector
    pub fn transform_vector(&self, v: &VectorInnerVec<T>) -> VectorInnerVec<T> {
        assert!(self.size() == v.len());

        let mut x = VectorInnerVec::new(vec![]);

        let n = self.size();
        for i in 0..n {
            let mut y = T::ZERO;

            for j in 0..n {
                y += self[(i, j)] * v[j];
            }

            x.push(y)
        }

        x
    }

    /// Solving a system of linear equations `A * x = b` by LU factorization
    pub fn lu_solve(&self, b: &VectorInnerVec<T>) -> VectorInnerVec<T> {
        assert!(self.size() == b.len());

        let (lt, ut) = self.lu();
        let x1 = lt.back_substitution(b);
        ut.back_substitution(&x1)
    }

    pub(crate) fn get_inner_vec(self) -> Vec<T> {
        self.inner.0
    }
}

impl<T> Square for MatrixSquareFullVec<T> {
    fn size(&self) -> usize {
        self.row_size()
    }
}

super::impl_index_usize_tuple!(MatrixSquareFullVec<T>);

impl<T: CoreFloat> From<MatrixLTVec<T>> for MatrixSquareFullVec<T> {
    fn from(lt: MatrixLTVec<T>) -> Self {
        let n = lt.size();
        let mut m = MatrixSquareFullVec::new_with_vec(n, vec_zeros(n * n));
        for i in 0..n {
            for j in 0..n {
                m[(i, j)] = lt[(i, j)];
            }
        }

        m
    }
}

impl<T: CoreFloat> From<MatrixUTVec<T>> for MatrixSquareFullVec<T> {
    fn from(lt: MatrixUTVec<T>) -> Self {
        let n = lt.size();
        let mut m = MatrixSquareFullVec::new_with_vec(n, vec_zeros(n * n));
        for i in 0..n {
            for j in 0..n {
                m[(i, j)] = lt[(i, j)];
            }
        }

        m
    }
}
