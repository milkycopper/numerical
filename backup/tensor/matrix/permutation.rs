use crate::tensor::vector::VectorInnerVec;

use super::{index2d::Index2D, Matrix, MatrixMulSelf, MatrixSquareFullVec, Square};
use core::ops::{Deref, DerefMut, Index, Mul};
use core_float::CoreFloat;

/// A permutation matrix is an `N * N` matrix consisting of all zeros, except for a single 1 in
/// every row and column. Equivalently, a permutation matrix P is created by applying arbitrary
/// row exchanges to the `N * N` identity matrix (or arbitrary column exchanges).
#[derive(Clone, Eq, PartialEq)]
pub struct PermutationVec<T> {
    permutation: Vec<usize>,
    zero: T,
    one: T,
}

impl<T> Deref for PermutationVec<T> {
    type Target = Vec<usize>;
    fn deref(&self) -> &Self::Target {
        &self.permutation
    }
}

impl<T> DerefMut for PermutationVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.permutation
    }
}

impl<T: CoreFloat> Index<Index2D> for Matrix<T, PermutationVec<T>> {
    type Output = T;

    fn index(&self, index: Index2D) -> &Self::Output {
        if self.deref()[index.row] == index.col {
            &self.one
        } else {
            &self.zero
        }
    }
}

/// Matrix whose inner storage type is `PermutationVec`
pub type MatrixPermutationVec<T> = Matrix<T, PermutationVec<T>>;

impl<T> Square for MatrixPermutationVec<T> {
    fn size(&self) -> usize {
        self.shape.row
    }
}

impl<T: CoreFloat> MatrixPermutationVec<T> {
    pub fn new_from_vec(v: Vec<usize>) -> Self {
        Self {
            shape: (v.len(), v.len()).into(),
            inner: PermutationVec {
                permutation: v,
                zero: T::ZERO,
                one: T::ONE,
            },
            phantom: std::marker::PhantomData,
        }
    }

    pub fn identity(n: usize) -> Self {
        Self::new_from_vec((0..n).collect())
    }

    pub fn exchange(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        };
        assert!(a < self.size());
        assert!(b < self.size());

        self.deref_mut().swap(a, b);
    }
}

impl<T: CoreFloat> Index<(usize, usize)> for MatrixPermutationVec<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self[Index2D::from(index)]
    }
}

impl<T: CoreFloat> MatrixMulSelf<T> for MatrixPermutationVec<T> {
    fn mul(&self, rhs: &Self) -> Self {
        assert!(self.size() == rhs.size());
        let mut p = Self::identity(self.size());
        for i in 0..self.size() {
            p.permutation[i] = rhs.permutation[self.deref()[i]];
        }

        p
    }
}

impl<T: CoreFloat> Mul<&MatrixSquareFullVec<T>> for MatrixPermutationVec<T> {
    type Output = MatrixSquareFullVec<T>;

    fn mul(self, rhs: &MatrixSquareFullVec<T>) -> Self::Output {
        assert!(self.size() == rhs.size());
        let mut m = rhs.clone();

        let n = self.size();
        for i in 0..n {
            if self.deref()[i] != i {
                for j in 0..n {
                    m[(i, j)] = rhs[(self.deref()[i], j)]
                }
            }
        }

        m
    }
}

impl<T: CoreFloat> Mul<&MatrixSquareFullVec<T>> for &MatrixPermutationVec<T> {
    type Output = MatrixSquareFullVec<T>;

    fn mul(self, rhs: &MatrixSquareFullVec<T>) -> Self::Output {
        assert!(self.size() == rhs.size());
        let mut m = rhs.clone();

        let n = self.size();
        for i in 0..n {
            if self.deref()[i] != i {
                for j in 0..n {
                    m[(i, j)] = rhs[(self.deref()[i], j)]
                }
            }
        }

        m
    }
}

impl<T: CoreFloat> Mul<&VectorInnerVec<T>> for MatrixPermutationVec<T> {
    type Output = VectorInnerVec<T>;

    fn mul(self, rhs: &VectorInnerVec<T>) -> Self::Output {
        assert!(self.size() == rhs.len());
        let mut v = (*rhs).clone();

        let n = self.size();
        for i in 0..n {
            if self.deref()[i] != i {
                v[i] = rhs[self.deref()[i]];
            }
        }

        VectorInnerVec::new(v)
    }
}

impl<T: CoreFloat> Mul<&VectorInnerVec<T>> for &MatrixPermutationVec<T> {
    type Output = VectorInnerVec<T>;

    fn mul(self, rhs: &VectorInnerVec<T>) -> Self::Output {
        assert!(self.size() == rhs.len());
        let mut v = (*rhs).clone();

        let n = self.size();
        for i in 0..n {
            if self.deref()[i] != i {
                v[i] = rhs[self.deref()[i]];
            }
        }

        VectorInnerVec::new(v)
    }
}
