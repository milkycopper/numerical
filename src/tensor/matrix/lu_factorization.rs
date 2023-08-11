use core_float::CoreFloat;

use super::{MatrixLTVec, MatrixSquareFullVec, MatrixUTVec, Square};

pub trait LUFactorization {
    type LowerTriangular;
    type UpperTriangular;

    fn lu(&self) -> (Self::LowerTriangular, Self::UpperTriangular);
}

impl<T: CoreFloat> LUFactorization for MatrixSquareFullVec<T> {
    type UpperTriangular = MatrixUTVec<T>;
    type LowerTriangular = MatrixLTVec<T>;

    fn lu(&self) -> (Self::LowerTriangular, Self::UpperTriangular) {
        let n = self.size();

        assert!(n > 1);

        let mut lower =
            Self::UpperTriangular::new_with_vec(self.size() - 1, vec![T::ZERO; n * (n - 1) / 2]);
        let mut upper =
            Self::UpperTriangular::new_with_vec(self.size(), vec![T::ZERO; n * (n + 1) / 2]);
        let mut m = self.clone();

        for j in 0..n {
            let e0 = m[(j, j)].recip();

            assert!(
                e0.is_finite(),
                "pivot element at ({}, {}) is 0, LU Factorization halt.",
                j,
                j
            );

            upper[(j, j)] = m[(j, j)];

            for i in (j + 1)..n {
                let e1 = m[(i, j)] * e0;
                upper[(j, i)] = m[(j, i)];
                lower[(j, i - 1)] = e1;
                m[(i, j)] = T::ZERO;

                for k in (j + 1)..n {
                    let x = m[(j, k)];
                    m[(i, k)] -= e1 * x;
                }
            }
        }

        let lower = lower
            .transpose()
            .extend_with_diagonal(&mut (0..n).map(|_| T::ONE));

        (lower, upper)
    }
}
