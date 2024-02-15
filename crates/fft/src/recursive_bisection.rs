use floating_point::Complex64;

use crate::DFT;

fn is_power_of_2(x: usize) -> bool {
    x > 1 && (x & (x - 1) == 0)
}

pub struct RecursiveBisectionFFT;

impl DFT for RecursiveBisectionFFT {
    fn fourier_transform(data: &Vec<Complex64>) -> Vec<Complex64> {
        assert!(!data.is_empty());

        let n = data.len();

        if n == 1 {
            data.clone()
        } else {
            assert!(is_power_of_2(n));

            let x1 = (0..(n / 2))
                .map(|i| data[i] + data[i + n / 2])
                .collect::<Vec<_>>();

            let x2 = (0..(n / 2))
                .map(|i| (data[i] - data[i + n / 2]) * Complex64::omega_n_power(n, i))
                .collect::<Vec<_>>();

            let y1 = RecursiveBisectionFFT::fourier_transform(&x1);
            let y2 = RecursiveBisectionFFT::fourier_transform(&x2);

            (0..(n / 2))
                .into_iter()
                .map(|i| [y1[i], y2[i]])
                .collect::<Vec<_>>()
                .concat()
        }
    }

    fn inverse_fourier_transform(data: &Vec<Complex64>) -> Vec<Complex64> {
        fn inverse_fourier_transform_recursive(
            data: &Vec<Complex64>,
            is_first_time: bool,
        ) -> Vec<Complex64> {
            assert!(!data.is_empty());

            let n = data.len();

            if n == 1 {
                data.clone()
            } else {
                assert!(is_power_of_2(n));

                let x1 = (0..(n / 2))
                    .map(|i| data[i] + data[i + n / 2])
                    .collect::<Vec<_>>();

                let x2 = (0..(n / 2))
                    .map(|i| {
                        (data[i] - data[i + n / 2]) * Complex64::omega_n_power(n, i).conjugate()
                    })
                    .collect::<Vec<_>>();

                let y1 = inverse_fourier_transform_recursive(&x1, false);
                let y2 = inverse_fourier_transform_recursive(&x2, false);

                (0..(n / 2))
                    .into_iter()
                    .map(|i| [y1[i], y2[i]])
                    .collect::<Vec<_>>()
                    .concat()
                    .into_iter()
                    .map(|x| {
                        if is_first_time {
                            x * Complex64::new(1.0 / n as f64, 0.0)
                        } else {
                            x
                        }
                    })
                    .collect::<Vec<_>>()
            }
        }
        inverse_fourier_transform_recursive(data, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_power_of_2() {
        assert!(!is_power_of_2(0));
        assert!(!is_power_of_2(1));
        assert!(!is_power_of_2(3));
        assert!(!is_power_of_2(7));

        assert!(is_power_of_2(2));
        assert!(is_power_of_2(4));
        assert!(is_power_of_2(8));
    }
}
