use core::ops::Index;
use std::{fmt::Display, ops::IndexMut};

use floating_point::F64;

use crate::Matrix;

#[derive(Debug, Clone, Copy)]
pub enum TriangleMatType {
    Upper,
    Lower,
}

#[derive(Debug, Clone)]
pub struct TriFullMat<T> {
    pub ty: TriangleMatType,
    n: usize,
    storage: Vec<T>,
}

impl<T> TriFullMat<T> {
    fn index_in_vec(&self, index: (usize, usize)) -> usize {
        let (i, j) = index;
        match self.ty {
            TriangleMatType::Upper => (2 * self.n - 1 - i) * i / 2 + j,
            TriangleMatType::Lower => (1 + i) * i / 2 + j,
        }
    }
}

impl<T: Clone> TriFullMat<T> {
    pub fn from_rows(ty: TriangleMatType, rows: Vec<Vec<T>>) -> Self {
        assert!(!rows.is_empty());

        let n = rows.len();
        for (i, row) in rows.iter().enumerate() {
            assert!(
                row.len()
                    == match ty {
                        TriangleMatType::Upper => n - i,
                        TriangleMatType::Lower => i + 1,
                    }
            );
        }

        Self {
            ty,
            n,
            storage: rows.concat(),
        }
    }

    pub fn from_vec(ty: TriangleMatType, elements: Vec<T>) -> Self {
        let mut left = elements.len();
        let mut n = 0;
        while left > n {
            left -= n + 1;
            n += 1;
        }
        assert!(left == 0);

        Self {
            ty,
            n,
            storage: elements,
        }
    }
}

impl TriFullMat<F64> {
    pub fn solve(&self, b: Vec<F64>) -> Vec<F64> {
        assert!(self.n == b.len());

        match self.ty {
            TriangleMatType::Lower => {
                let mut x = vec![];
                for i in 0..self.col_count() {
                    let mut sum: F64 = 0.0.into();
                    for j in 0..i {
                        sum += x[j] * self[(i, j)];
                    }
                    x.push((b[i] - sum) / self[(i, i)]);
                }

                x
            }
            TriangleMatType::Upper => {
                let mut x = vec![F64::from(0.0); self.n];
                for i in (0..self.n).rev() {
                    let mut sum: F64 = 0.0.into();
                    for j in (i + 1)..self.n {
                        sum += x[j] * self[(i, j)];
                    }
                    x[i] = (b[i] - sum) / self[(i, i)];
                }

                x
            }
        }
    }
}

impl Index<(usize, usize)> for TriFullMat<F64> {
    type Output = F64;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (i, j) = index;
        if let TriangleMatType::Upper = self.ty {
            if i > j {
                return &F64::ZERO;
            }
        }
        if let TriangleMatType::Lower = self.ty {
            if i < j {
                return &F64::ZERO;
            }
        }
        &self.storage[self.index_in_vec(index)]
    }
}

impl IndexMut<(usize, usize)> for TriFullMat<F64> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (i, j) = index;
        if let TriangleMatType::Upper = self.ty {
            assert!(i < j);
        }
        if let TriangleMatType::Lower = self.ty {
            assert!(i > j);
        }
        let i = self.index_in_vec(index);
        &mut self.storage[i]
    }
}

impl Matrix<F64> for TriFullMat<F64> {
    fn shape(&self) -> (usize, usize) {
        (self.n, self.n)
    }
}

impl Display for TriFullMat<F64> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Matrix::fmt(self, f)
    }
}
