use core::cmp::Ordering;

use crate::base_float::BaseFloat;

use super::{MatrixLTVec, MatrixPermutationVec, MatrixSquareFullVec, MatrixUTVec, Square};

pub trait LUFactorization {
    type LowerTriangular;
    type UpperTriangular;
    type Permutation;

    fn lu(&self) -> (Self::LowerTriangular, Self::UpperTriangular);
    fn pa_lu(
        &self,
    ) -> (
        Self::Permutation,
        Self::LowerTriangular,
        Self::UpperTriangular,
    );
}

impl<T: BaseFloat> LUFactorization for MatrixSquareFullVec<T> {
    type UpperTriangular = MatrixUTVec<T>;
    type LowerTriangular = MatrixLTVec<T>;
    type Permutation = MatrixPermutationVec<T>;

    fn lu(&self) -> (Self::LowerTriangular, Self::UpperTriangular) {
        let n = self.size();

        assert!(n > 1);

        let mut m = self.clone();

        for j in 0..n {
            let e0 = m[(j, j)].recip();

            assert!(
                e0.is_finite(),
                "pivot element at ({}, {}) is 0, LU Factorization halt.",
                j,
                j
            );

            for i in (j + 1)..n {
                let e1 = m[(i, j)] * e0;
                m[(i, j)] = e1;

                for k in (j + 1)..n {
                    let x = m[(j, k)];
                    m[(i, k)] -= e1 * x;
                }
            }
        }

        let mut lower_vec = vec![];
        let mut upper_vec = vec![];

        for i in 0..n {
            for j in 0..n {
                match i.cmp(&j) {
                    Ordering::Greater => lower_vec.push(m[(i, j)]),
                    Ordering::Equal => {
                        lower_vec.push(T::ONE);
                        upper_vec.push(m[(i, j)]);
                    }
                    Ordering::Less => upper_vec.push(m[(i, j)]),
                }
            }
        }

        (
            MatrixLTVec::new_with_vec(n, lower_vec),
            MatrixUTVec::new_with_vec(n, upper_vec),
        )
    }

    fn pa_lu(
        &self,
    ) -> (
        Self::Permutation,
        Self::LowerTriangular,
        Self::UpperTriangular,
    ) {
        let n = self.size();

        assert!(n > 1);

        let mut permutation = MatrixPermutationVec::<T>::identity(n);
        let mut m = self.clone();

        for j in 0..n {
            let (mut max_index, mut max) = (j, m[(j, j)].abs());
            for i in (j + 1)..n {
                if m[(i, j)].abs() > max {
                    max_index = i;
                    max = m[(i, j)].abs();
                }
            }

            if max_index != j {
                m.exchange_row(j, max_index);
                permutation.exchange(j, max_index);
            }

            let e0 = m[(j, j)].recip();

            assert!(
                e0.is_finite(),
                "pivot element at ({}, {}) is 0, LU Factorization halt.",
                j,
                j
            );

            for i in (j + 1)..n {
                let e1 = m[(i, j)] * e0;
                m[(i, j)] = e1;

                for k in (j + 1)..n {
                    let x = m[(j, k)];
                    m[(i, k)] -= e1 * x;
                }
            }
        }

        let mut lower_vec = vec![];
        let mut upper_vec = vec![];

        for i in 0..n {
            for j in 0..n {
                match i.cmp(&j) {
                    Ordering::Greater => lower_vec.push(m[(i, j)]),
                    Ordering::Equal => {
                        lower_vec.push(T::ONE);
                        upper_vec.push(m[(i, j)]);
                    }
                    Ordering::Less => upper_vec.push(m[(i, j)]),
                }
            }
        }

        (
            permutation,
            MatrixLTVec::new_with_vec(n, lower_vec),
            MatrixUTVec::new_with_vec(n, upper_vec),
        )
    }
}
