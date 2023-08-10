mod base_traits;
mod full_inner_vec;
mod index2d;
mod lt_inner_vec;
mod ut_inner_vec;

pub use base_traits::{MatrixBaseOps, Square};
pub use full_inner_vec::MatrixFullInnerVec;
pub use index2d::Index2D;
pub use lt_inner_vec::MatrixLTVec;
pub use ut_inner_vec::MatrixUTVec;

use base_traits::impl_index_usize_tuple;
use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};

/// 2D matrix wrapper type
///
/// # Generic
/// - T - Element type
/// - S - Inner concrete type
#[derive(Clone, PartialEq, Eq)]
pub struct Matrix<T, S> {
    shape: Index2D,
    inner: S,
    phantom: PhantomData<T>,
}

impl<T, S> Matrix<T, S> {
    /// Matrix constructor
    ///
    /// # Arguments
    ///
    /// * `shape` - The 2D shape of matrix
    /// * `inner` - inner type instance
    pub const fn new(shape: Index2D, inner: S) -> Self {
        Self {
            shape,
            inner,
            phantom: PhantomData,
        }
    }

    pub const fn row_size(&self) -> usize {
        self.shape.row
    }

    pub const fn col_size(&self) -> usize {
        self.shape.col
    }

    pub const fn shape(&self) -> Index2D {
        Index2D {
            row: self.row_size(),
            col: self.col_size(),
        }
    }

    pub fn shape_eq<S1>(&self, rhs: Matrix<T, S1>) -> bool {
        self.shape() == rhs.shape()
    }
}

impl<T, S> Deref for Matrix<T, S> {
    type Target = S;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T, S> DerefMut for Matrix<T, S> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}
