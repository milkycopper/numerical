use core::ops::{Index, IndexMut};

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
}

impl<T: Copy> FullMat<T> {
    pub fn swap(&mut self, index_a: (u32, u32), index_b: (u32, u32)) {
        let temp = self[index_a];
        self[index_a] = self[index_b];
        self[index_b] = temp;
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

impl<T> Matrix<T> for FullMat<T> {
    fn shape(&self) -> (u32, u32) {
        assert!(self.storage.len() as u32 % self.col_count == 0);
        (self.storage.len() as u32 / self.col_count, self.col_count)
    }
}

impl FullMat<F64> {
    pub fn lu(&self) -> (Vec<u32>, FullMat<F64>, FullMat<F64>) {
        let mut mat = self.clone();

        assert!(mat.is_square());
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
                for k in 0..n {
                    mat.swap((j, k), (max_abs_row, k));
                }
                p.swap(j as usize, max_abs_row as usize);
            }

            for i in (j + 1)..n {
                let pivot = mat[(j, j)];
                let m = mat[(i, j)] / pivot;
                mat[(i, j)] = m;
                for k in (j + 1)..n {
                    let pivot = mat[(j, k)];
                    mat[(i, k)] -= m * pivot;
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

        (p, l, u)
    }
}
