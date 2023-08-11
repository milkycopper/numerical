use super::{index2d::Index2D, Matrix, MatrixBaseOps, MatrixLTVec, MatrixUTVec, Square};
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

impl<T: CoreFloat> MatrixBaseOps<T> for Matrix<T, SquareFullVec<T>> {
    fn add(&self, rhs: &Self) -> Self {
        let mut inner = vec![T::ZERO; self.shape.area_size()];
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
        let mut inner = vec![T::ZERO; self.shape.area_size()];
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

    fn mul(&self, rhs: &Self) -> Self {
        assert!(self.size() == rhs.size());
        let mut mat = Self::new_with_vec(self.size(), vec![T::ZERO; self.size() * self.size()]);
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
        let mut m = Self::new_with_vec(self.size(), vec![T::ZERO; self.shape.area_size()]);
        for j in 0..self.size() {
            for i in 0..self.size() {
                m[(j, i)] = self[(i, j)];
            }
        }

        m
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
        let mut m = MatrixSquareFullVec::new_with_vec(n, vec![T::ZERO; n * n]);
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
        let mut m = MatrixSquareFullVec::new_with_vec(n, vec![T::ZERO; n * n]);
        for i in 0..n {
            for j in 0..n {
                m[(i, j)] = lt[(i, j)];
            }
        }

        m
    }
}
