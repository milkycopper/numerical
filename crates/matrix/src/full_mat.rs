use core::ops::{Index, IndexMut};
use std::fmt::{Debug, Display};

use floating_point::F64;

use crate::Matrix;

#[derive(Clone, Debug)]
pub struct FullMat<T> {
    col_count: u32,
    storage: Vec<T>,
}

impl<T> FullMat<T> {
    fn pos(&self, index: (u32, u32)) -> usize {
        (index.0 * self.col_count + index.1) as usize
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
        let storage: Vec<T> = rows.concat();
        Self {
            col_count: col_count as u32,
            storage,
        }
    }

    pub fn from_vec(col_count: u32, elements: Vec<T>) -> Self {
        assert!(col_count > 0);
        assert!(!elements.is_empty());
        assert!(elements.len() as u32 % col_count == 0);
        Self {
            col_count,
            storage: elements,
        }
    }
}

impl<T: Copy> FullMat<T> {
    pub fn swap(&mut self, index_a: (u32, u32), index_b: (u32, u32)) {
        let pa = self.pos(index_a);
        let pb = self.pos(index_b);
        self.storage.swap(pa, pb)
    }

    pub fn swap_row(&mut self, row_a: u32, row_b: u32) {
        if row_a != row_b {
            let min_row = row_a.min(row_b);
            let max_row = row_a.max(row_b);
            let left_count = (min_row + 1) * self.col_count;
            let (left, right) = self.storage.split_at_mut(left_count as usize);
            let left_start = (min_row * self.col_count) as usize;
            let right_start = ((max_row - min_row - 1) * self.col_count) as usize;
            left[left_start..]
                .swap_with_slice(&mut right[right_start..(right_start + self.col_count as usize)])
        }
    }
}

impl<T> Index<(u32, u32)> for FullMat<T> {
    type Output = T;
    fn index(&self, index: (u32, u32)) -> &Self::Output {
        &self.storage[self.pos(index)]
    }
}

impl<T> IndexMut<(u32, u32)> for FullMat<T> {
    fn index_mut(&mut self, index: (u32, u32)) -> &mut Self::Output {
        let pos = self.pos(index);
        &mut self.storage[pos]
    }
}

impl<T: Display> Matrix<T> for FullMat<T> {
    fn shape(&self) -> (u32, u32) {
        debug_assert!(self.storage.len() as u32 % self.col_count == 0);
        (self.storage.len() as u32 / self.col_count, self.col_count)
    }
}

impl<T: Display> Display for FullMat<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Matrix::fmt(self, f)
    }
}

impl FullMat<F64> {
    pub fn lu(&self) -> Option<(FullMat<F64>, FullMat<F64>, Vec<u32>)> {
        assert!(self.is_square());

        let mut mat = self.clone();
        let n = mat.row_count();
        let mut p: Vec<u32> = (0..n).collect();

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
                p.swap(j as usize, max_abs_row as usize);
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

        let mut l = mat.clone();
        let mut u = mat.clone();

        for i in 0..n {
            l[(i, i)] = 1.0.into();
            for j in (i + 1)..n {
                l[(i, j)] = 0.0.into();
            }
            for j in 0..i {
                u[(i, j)] = 0.0.into();
            }
        }

        Some((l, u, p))
    }

    pub fn lower_tri_back_substitution(&self, b: Vec<F64>) -> Vec<F64> {
        assert!(self.is_square());
        assert!(b.len() as u32 == self.col_count);
        let n = self.col_count;
        let mut y = vec![];
        for i in 0..n {
            let mut sum: F64 = 0.0.into();
            for j in 0..i {
                sum += y[j as usize] * self[(i, j)];
            }
            y.push((b[i as usize] - sum) / self[(i, i)]);
        }

        y
    }

    pub fn upper_tri_back_substitution(&self, b: Vec<F64>) -> Vec<F64> {
        assert!(self.is_square());
        assert!(b.len() as u32 == self.col_count);
        let n = self.col_count;
        let mut y = vec![F64::from(0.0); n as usize];
        for i in (0..n).rev() {
            let mut sum: F64 = 0.0.into();
            for j in (i + 1)..n {
                sum += y[j as usize] * self[(i, j)];
            }
            y[i as usize] = (b[i as usize] - sum) / self[(i, i)];
        }

        y
    }

    pub fn lu_solve(&self, b: Vec<F64>) -> Option<Vec<F64>> {
        assert!(self.is_square());
        assert!(b.len() as u32 == self.col_count);

        if let Some((l, u, p)) = self.lu() {
            let pb: Vec<F64> = p.into_iter().map(|i| b[i as usize]).collect();
            let y = l.lower_tri_back_substitution(pb);
            let y = u.upper_tri_back_substitution(y);
            Some(y)
        } else {
            None
        }
    }
}
