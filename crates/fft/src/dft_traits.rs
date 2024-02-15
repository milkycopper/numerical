use floating_point::Complex64;

pub trait DFT {
    fn fourier_transform(data: &Vec<Complex64>) -> Vec<Complex64>;
    fn inverse_fourier_transform(data: &Vec<Complex64>) -> Vec<Complex64>;
}
