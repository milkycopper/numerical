use floating_point::Complex64;

use crate::DFT;

pub struct NaiveDFT;

impl DFT for NaiveDFT {
    fn fourier_transform(data: &[Complex64]) -> Vec<Complex64> {
        let n = data.len();
        (0..n)
            .map(|k| {
                (0..n)
                    .map(|j| data[j] * Complex64::omega_n_power(n, j * k))
                    .sum()
            })
            .collect::<Vec<_>>()
    }

    fn inverse_fourier_transform(data: &[Complex64]) -> Vec<Complex64> {
        let n = data.len();
        (0..n)
            .map(|k| {
                (0..n)
                    .map(|j| data[j] * Complex64::omega_n_power(n, j * k).conjugate())
                    .sum::<Complex64>()
                    * Complex64::new(1.0 / n as f64, 0.0)
            })
            .collect::<Vec<_>>()
    }
}
