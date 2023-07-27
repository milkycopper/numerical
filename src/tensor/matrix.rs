use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Index, IndexMut, Mul, Sub, SubAssign};
use core_float::core_float_traits::CoreFloat;
use std::fmt::Display;

pub mod inner_vec;
pub use inner_vec::MatrixInnerVec;

/// Index for 2D matrix
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Index2D {
    pub row: usize,
    pub col: usize,
}

impl Index2D {
    pub fn to_1d(&self, col_size: usize) -> usize {
        self.row * col_size + self.col
    }
}

impl From<(usize, usize)> for Index2D {
    fn from(value: (usize, usize)) -> Self {
        Index2D {
            row: value.0,
            col: value.1,
        }
    }
}

pub trait MatrixShape {
    fn row_size(&self) -> usize;
    fn col_size(&self) -> usize;
    fn square_dimension(&self) -> Option<usize> {
        if self.row_size() == self.col_size() {
            Some(self.row_size())
        } else {
            None
        }
    }
    fn shape(&self) -> Index2D {
        Index2D {
            row: self.row_size(),
            col: self.col_size(),
        }
    }
}

/// Operations for matrix
pub trait MatrixOps<T: CoreFloat>:
    MatrixShape + Index<Index2D, Output = T> + IndexMut<Index2D> + Clone
{
    fn default_with_shape(shape: Index2D) -> Self;

    fn shape_eq(&self, rhs: &impl MatrixOps<T>) -> bool {
        self.shape() == rhs.shape()
    }

    fn add(&self, rhs: &Self) -> Self {
        assert!(self.shape_eq(rhs));
        let mut mat = Self::default_with_shape(self.shape());
        for row in 0..self.row_size() {
            for col in 0..self.col_size() {
                mat[(row, col).into()] = self[(row, col).into()] + rhs[(row, col).into()]
            }
        }

        mat
    }

    fn add_assign(&mut self, rhs: &Self) {
        assert!(self.shape_eq(rhs));
        for row in 0..self.row_size() {
            for col in 0..self.col_size() {
                self[(row, col).into()] += rhs[(row, col).into()]
            }
        }
    }

    fn sub(&self, rhs: &Self) -> Self {
        assert!(self.shape_eq(rhs));
        let mut mat = Self::default_with_shape(self.shape());
        for row in 0..self.row_size() {
            for col in 0..self.col_size() {
                mat[(row, col).into()] = self[(row, col).into()] - rhs[(row, col).into()]
            }
        }

        mat
    }

    fn sub_assign(&mut self, rhs: &Self) {
        assert!(self.shape_eq(rhs));
        for row in 0..self.row_size() {
            for col in 0..self.col_size() {
                self[(row, col).into()] -= rhs[(row, col).into()]
            }
        }
    }

    fn mul(&self, rhs: &impl MatrixOps<T>) -> Self {
        assert!(self.col_size() == rhs.row_size());
        let mut mat = Self::default_with_shape((self.row_size(), rhs.col_size()).into());
        for row in 0..self.row_size() {
            for col in 0..rhs.col_size() {
                for i in 0..self.col_size() {
                    mat[(row, col).into()] += self[(row, i).into()] * self[(i, col).into()];
                }
            }
        }

        mat
    }
}

/// 2D matrix represented by its shape and inner storage
///
/// # Generic
/// - T - Element type
/// - S - Inner storage type
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
    /// * `storage` - storage of matrix elements
    pub fn new(shape: Index2D, storage: S) -> Self {
        Self {
            shape,
            inner: storage,
            phantom: PhantomData,
        }
    }
}

impl<T, S> MatrixShape for Matrix<T, S> {
    fn row_size(&self) -> usize {
        self.shape.row
    }
    fn col_size(&self) -> usize {
        self.shape.col
    }
}

impl<T, S> Add for Matrix<T, S>
where
    T: CoreFloat,
    Matrix<T, S>: MatrixOps<T>,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        <Self as MatrixOps<T>>::add(&self, &rhs)
    }
}

impl<T, S> Add<&Self> for Matrix<T, S>
where
    T: CoreFloat,
    Matrix<T, S>: MatrixOps<T>,
{
    type Output = Self;
    fn add(self, rhs: &Self) -> Self::Output {
        <Self as MatrixOps<T>>::add(&self, rhs)
    }
}

impl<T, S> AddAssign for Matrix<T, S>
where
    T: CoreFloat,
    Matrix<T, S>: MatrixOps<T>,
{
    fn add_assign(&mut self, rhs: Self) {
        <Self as MatrixOps<T>>::add_assign(self, &rhs);
    }
}

impl<T, S> AddAssign<&Self> for Matrix<T, S>
where
    T: CoreFloat,
    Matrix<T, S>: MatrixOps<T>,
{
    fn add_assign(&mut self, rhs: &Self) {
        <Self as MatrixOps<T>>::add_assign(self, rhs);
    }
}

impl<T, S> Sub for Matrix<T, S>
where
    T: CoreFloat,
    Matrix<T, S>: MatrixOps<T>,
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        <Self as MatrixOps<T>>::sub(&self, &rhs)
    }
}

impl<T, S> Sub<&Self> for Matrix<T, S>
where
    T: CoreFloat,
    Matrix<T, S>: MatrixOps<T>,
{
    type Output = Self;
    fn sub(self, rhs: &Self) -> Self::Output {
        <Self as MatrixOps<T>>::sub(&self, rhs)
    }
}

impl<T, S> SubAssign for Matrix<T, S>
where
    T: CoreFloat,
    Matrix<T, S>: MatrixOps<T>,
{
    fn sub_assign(&mut self, rhs: Self) {
        <Self as MatrixOps<T>>::sub_assign(self, &rhs);
    }
}

impl<T, S> SubAssign<&Self> for Matrix<T, S>
where
    T: CoreFloat,
    Matrix<T, S>: MatrixOps<T>,
{
    fn sub_assign(&mut self, rhs: &Self) {
        <Self as MatrixOps<T>>::sub_assign(self, rhs);
    }
}

impl<T, S, Rhs> Mul<Rhs> for Matrix<T, S>
where
    Rhs: MatrixOps<T>,
    T: CoreFloat,
    Matrix<T, S>: MatrixOps<T>,
{
    type Output = Self;
    fn mul(self, rhs: Rhs) -> Self::Output {
        <Self as MatrixOps<T>>::mul(&self, &rhs)
    }
}

impl<T, S> Display for Matrix<T, S>
where
    T: CoreFloat,
    Matrix<T, S>: MatrixOps<T>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "[")?;
        for i in 0..self.row_size() {
            write!(f, "[")?;
            for j in 0..self.col_size() {
                write!(f, "{}", self[(i, j).into()])?;
                if j != self.col_size() - 1 {
                    write!(f, ", ")?;
                }
            }
            write!(f, "]")?;
            if i != self.row_size() - 1 {
                writeln!(f, ",")?;
            }
        }
        write!(f, "\n]")?;

        Ok(())
    }
}
