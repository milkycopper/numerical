use core::marker::PhantomData;
use core::ops::{Deref, DerefMut, Index, IndexMut};

use approx::AbsDiffEq;
use core_float::CoreFloat;

macro_rules! impl_vec_storage_for_array {
    ( $( $n:literal ),* ) => {
        $(
            impl<T> LinearStorageLen for [T; $n] {
                #[inline]
                fn len(&self) -> usize {
                    $n
                }
                #[inline]
                fn is_empty(&self) -> bool {
                    $n == 0
                }
            }

            impl<T> VecStorage<T> for [T; $n] {}
        )*
    };
}

pub trait LinearStorageLen {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
}

pub trait VecStorage<T>: LinearStorageLen + IndexMut<usize> + Index<usize, Output = T> {}

impl_vec_storage_for_array!(
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32
);

impl<T> LinearStorageLen for Vec<T> {
    #[inline]
    fn len(&self) -> usize {
        self.len()
    }
    #[inline]
    fn is_empty(&self) -> bool {
        self.is_empty()
    }
}

/// Vector in mathematical sense whose inner storage type is [`Vec`]
impl<T> VecStorage<T> for Vec<T> {}

/// Vector in mathematical sense
///
/// # Generic
/// * T: element type
/// * S: storage type (Array, Vec etc.)
#[derive(PartialEq, Debug)]
pub struct Vector<T, S: VecStorage<T>> {
    inner: S,
    phantom: PhantomData<T>,
}

pub type VectorInnerVec<T> = Vector<T, Vec<T>>;

impl<T, S: VecStorage<T>> Vector<T, S> {
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            phantom: PhantomData,
        }
    }
}

impl<T, S: VecStorage<T>> Deref for Vector<T, S> {
    type Target = S;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T, S: VecStorage<T>> DerefMut for Vector<T, S> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<T, S: VecStorage<T>> From<S> for Vector<T, S> {
    fn from(value: S) -> Self {
        Vector::new(value)
    }
}

impl<T: CoreFloat + AbsDiffEq<Epsilon = T>> approx::AbsDiffEq for VectorInnerVec<T> {
    type Epsilon = T;

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        if self.len() != other.len() {
            return false;
        }

        for (a, b) in self.iter().zip(other.iter()) {
            if !a.abs_diff_eq(b, epsilon) {
                return false;
            }
        }

        true
    }

    fn default_epsilon() -> Self::Epsilon {
        T::EPSILON
    }
}
