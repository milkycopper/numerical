use core::ops::{Index, IndexMut};
use std::fmt::{Debug, Display};

use floating_point::F64;

use crate::{Matrix, TriFullMat, TriangleMatType};

#[derive(Clone, Debug)]
pub struct FullMat<T> {
    storage: Vec<T>,
    col_count: usize,
}

impl<T> FullMat<T> {
    fn index_in_vec(&self, index: (usize, usize)) -> usize {
        index.0 * self.col_count + index.1
    }
}

impl<T: Clone> FullMat<T> {
    pub fn from_rows(rows: Vec<Vec<T>>) -> Self {
        assert!(!rows.is_empty());
        assert!(!rows[0].is_empty());

        let col_count = rows[0].len();
        for row in &rows[1..] {
            assert!(row.len() == col_count);
        }

        Self {
            col_count,
            storage: rows.concat(),
        }
    }

    pub fn from_vec(col_count: usize, elements: Vec<T>) -> Self {
        assert!(col_count > 0);
        assert!(!elements.is_empty());
        assert!(elements.len() % col_count == 0);
        Self {
            col_count,
            storage: elements,
        }
    }
}

impl<T: Copy> FullMat<T> {
    pub fn swap(&mut self, index_a: (usize, usize), index_b: (usize, usize)) {
        let ia = self.index_in_vec(index_a);
        let ib = self.index_in_vec(index_b);
        self.storage.swap(ia, ib)
    }

    pub fn swap_row(&mut self, row_a: usize, row_b: usize) {
        if row_a != row_b {
            let col_count = self.col_count;
            let min_row = row_a.min(row_b);
            let max_row = row_a.max(row_b);
            let left_count = (min_row + 1) * col_count;
            let (left, right) = self.storage.split_at_mut(left_count);
            let left_start = min_row * col_count;
            let right_start = (max_row - min_row - 1) * col_count;
            left[left_start..].swap_with_slice(&mut right[right_start..(right_start + col_count)])
        }
    }
}

impl FullMat<F64> {
    pub fn mul_mat(&self, rhs: &FullMat<F64>) -> FullMat<F64> {
        assert!(rhs.row_count() == self.col_count());

        let mut v = vec![];
        for i in 0..self.row_count() {
            for j in 0..rhs.col_count() {
                let mut elem = F64::from(0.0);
                for k in 0..self.col_count() {
                    elem += self[(i, k)] * rhs[(k, j)];
                }
                v.push(elem);
            }
        }

        FullMat::from_vec(rhs.col_count(), v)
    }

    pub fn mul_vec(&self, rhs: &Vec<F64>) -> Vec<F64> {
        assert!(self.col_count() == rhs.len());

        let mut v = vec![];
        for i in 0..self.row_count() {
            let mut elem = 0.0.into();
            for j in 0..self.col_count() {
                elem += self[(i, j)] * rhs[j];
            }
            v.push(elem);
        }

        v
    }

    pub fn add(&self, rhs: &FullMat<F64>) -> Self {
        assert!(self.shape() == rhs.shape());

        let v = self
            .storage
            .iter()
            .zip(rhs.storage.iter())
            .map(|(x, y)| x + y)
            .collect();

        Self::from_vec(self.col_count(), v)
    }

    pub fn sub(&self, rhs: &FullMat<F64>) -> Self {
        assert!(self.shape() == rhs.shape());

        let v = self
            .storage
            .iter()
            .zip(rhs.storage.iter())
            .map(|(x, y)| x - y)
            .collect();

        Self::from_vec(self.col_count(), v)
    }

    pub fn element_max_abs(&self) -> F64 {
        self.storage.iter().fold(0.0.into(), |max, x| max.max(*x))
    }

    pub fn lu(&self) -> Option<(TriFullMat<F64>, TriFullMat<F64>, Vec<usize>)> {
        assert!(self.is_square());

        let mut mat = self.clone();
        let n = mat.col_count();
        let mut p: Vec<usize> = (0..n).collect();

        for j in 0..n {
            let (mut max_abs, mut max_abs_row) = (mat[(j, j)].abs(), j);
            for i in (j + 1)..n {
                let abs = mat[(i, j)].abs();
                if abs > max_abs {
                    max_abs = abs;
                    max_abs_row = i;
                }
            }
            if max_abs_row != j {
                mat.swap_row(j, max_abs_row);
                p.swap(j, max_abs_row);
            }

            let pivot = mat[(j, j)];
            // the matrix is singular
            if pivot == 0.0.into() {
                return None;
            }
            for i in (j + 1)..n {
                let m = mat[(i, j)] / pivot;
                mat[(i, j)] = m;
                for k in (j + 1)..n {
                    let col_pivot = mat[(j, k)];
                    mat[(i, k)] -= m * col_pivot;
                }
            }
        }

        let mut l = vec![];
        let mut u = vec![];

        for i in 0..n {
            for j in 0..i {
                l.push(mat[(i, j)]);
            }
            l.push(1.0.into());
            for j in i..n {
                u.push(mat[(i, j)]);
            }
        }

        Some((
            TriFullMat::from_vec(TriangleMatType::Lower, l),
            TriFullMat::from_vec(TriangleMatType::Upper, u),
            p,
        ))
    }

    pub fn lu_solve(&self, b: &Vec<F64>) -> Option<Vec<F64>> {
        assert!(self.is_square());
        assert!(b.len() == self.col_count());

        if let Some((l, u, p)) = self.lu() {
            let pb: Vec<F64> = p.into_iter().map(|i| b[i]).collect();
            let y = l.solve(pb);
            let y = u.solve(y);
            Some(y)
        } else {
            None
        }
    }

    pub fn inv(&self) -> Option<Self> {
        assert!(self.is_square());

        if let Some((l, u, p)) = self.lu() {
            let mut cols = vec![];

            for i in 0..self.col_count() {
                let mut b = vec![F64::from(0.0); self.col_count()];
                b[i] = 1.0.into();
                let pb: Vec<F64> = p.iter().map(|i| b[*i]).collect();
                let y = l.solve(pb);
                let y = u.solve(y);
                cols.push(y);
            }

            let mut v = vec![];
            for i in 0..self.col_count() {
                for col in &cols {
                    v.push(col[i]);
                }
            }

            Some(Self::from_vec(self.col_count(), v))
        } else {
            None
        }
    }

    pub fn norm(&self) -> F64 {
        assert!(self.is_square());

        let mut col_sums = vec![];
        for j in 0..self.col_count() {
            let mut sum = 0.0.into();
            for i in 0..self.col_count() {
                sum += self[(i, j)].abs();
            }
            col_sums.push(sum);
        }

        col_sums.into_iter().fold(0.0.into(), |max, x| max.max(x))
    }
}

impl<T> Index<(usize, usize)> for FullMat<T> {
    type Output = T;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.storage[self.index_in_vec(index)]
    }
}

impl<T> IndexMut<(usize, usize)> for FullMat<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let i = self.index_in_vec(index);
        &mut self.storage[i]
    }
}

impl<T: Display> Matrix<T> for FullMat<T> {
    fn shape(&self) -> (usize, usize) {
        debug_assert!(self.storage.len() % self.col_count == 0);
        (self.storage.len() / self.col_count, self.col_count)
    }

    fn col_count(&self) -> usize {
        self.col_count
    }

    fn row_count(&self) -> usize {
        self.storage.len() / self.col_count()
    }
}

impl<T: Display> Display for FullMat<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Matrix::fmt(self, f)
    }
}

impl From<TriFullMat<F64>> for FullMat<F64> {
    fn from(tri_mat: TriFullMat<F64>) -> Self {
        let mut v = vec![];
        let n = tri_mat.col_count();
        match tri_mat.ty {
            TriangleMatType::Upper => {
                for i in 0..n {
                    for _ in 0..i {
                        v.push(0.0.into());
                    }
                    for j in i..n {
                        v.push(tri_mat[(i, j)])
                    }
                }
            }
            TriangleMatType::Lower => {
                for i in 0..n {
                    for j in 0..=i {
                        v.push(tri_mat[(i, j)]);
                    }
                    for _ in (i + 1)..n {
                        v.push(0.0.into())
                    }
                }
            }
        }
        Self::from_vec(n, v)
    }
}
