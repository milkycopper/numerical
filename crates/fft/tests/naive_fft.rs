use approx::AbsDiffEq;
use fft::{NaiveDFT, DFT};
use floating_point::Complex64;
use rand::prelude::*;

#[test]
fn test_naive_fft() {
    let mut rng = rand::thread_rng();
    let n = 1000;
    let x = (0..n)
        .into_iter()
        .map(|_| -> Complex64 { Complex64::new(rng.gen(), 0.0) })
        .collect::<Vec<_>>();
    //println!("x data is {:#?}", x);
    let y = NaiveDFT::fourier_transform(&x);
    //println!("y data is {:#?}", y);
    let z = NaiveDFT::inverse_fourier_transform(&y);
    //println!("z data is {:#?}", z);
    x.into_iter().zip(z.into_iter()).for_each(|(a, b)| {
        assert!(
            a.abs_diff_eq(&b, (2_f64.powi(7) * f64::EPSILON).into()),
            "a = {}, b = {}, diff = {}",
            a,
            b,
            a - b
        )
    });
}
