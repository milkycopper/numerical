#![cfg_attr(feature = "nightly_bench", feature(test))]

use approx::AbsDiffEq;
use fft::{NaiveDFT, RecursiveBisectionFFT, DFT};
use floating_point::Complex64;
use rand::prelude::*;

#[test]
fn test_recursive_bisection_fft_0() {
    let mut rng = rand::thread_rng();
    let n = 2_usize.pow(16);
    println!("n = {n}");
    let x = (0..n)
        .into_iter()
        .map(|_| -> Complex64 { Complex64::new(rng.gen(), 0.0) })
        .collect::<Vec<_>>();
    // println!("x data is {:#?}", x);
    let y = RecursiveBisectionFFT::fourier_transform(&x);
    // println!("y data is {:#?}", y);
    let z = RecursiveBisectionFFT::inverse_fourier_transform(&y);
    // println!("z data is {:#?}", z);
    x.into_iter().zip(z.into_iter()).for_each(|(a, b)| {
        assert!(
            a.abs_diff_eq(&b, (2_f64.powi(4) * f64::EPSILON).into()),
            "a = {}, b = {}, diff = {}",
            a,
            b,
            a - b
        )
    });
}

#[test]
fn test_recursive_bisection_fft_1() {
    let mut rng = rand::thread_rng();
    let n = 2_usize.pow(11);
    println!("n = {n}");
    let x = (0..n)
        .into_iter()
        .map(|_| -> Complex64 { Complex64::new(rng.gen(), 0.0) })
        .collect::<Vec<_>>();
    // println!("x data is {:#?}", x);
    let y1 = NaiveDFT::fourier_transform(&x);
    // println!("y1 data is {:#?}", y1);
    let y2 = RecursiveBisectionFFT::fourier_transform(&x);
    // println!("y2 data is {:#?}", y2);
    y1.into_iter().zip(y2.into_iter()).for_each(|(a, b)| {
        assert!(
            a.abs_diff_eq(&b, (1e4 * f64::EPSILON).into()),
            "a = {}, b = {}, diff = {:e}",
            a,
            b,
            a - b
        )
    });
}

#[cfg(feature = "nightly_bench")]
mod benchs {
    extern crate test;

    use super::*;

    fn prepare_data() -> Vec<Complex64> {
        let mut rng = rand::thread_rng();
        let n = 2_usize.pow(10);
        println!("n = {n}");
        (0..n)
            .into_iter()
            .map(|_| -> Complex64 { Complex64::new(rng.gen(), 0.0) })
            .collect::<Vec<_>>()
    }

    #[bench]
    fn bench_naive_fft(b: &mut test::Bencher) {
        let data = prepare_data();
        b.iter(|| NaiveDFT::fourier_transform(&data));
    }

    #[bench]
    fn bench_naive_inverse_fft(b: &mut test::Bencher) {
        let data = prepare_data();
        b.iter(|| NaiveDFT::inverse_fourier_transform(&data));
    }

    #[bench]
    fn bench_recursive_bisection_fft(b: &mut test::Bencher) {
        let data = prepare_data();
        b.iter(|| RecursiveBisectionFFT::fourier_transform(&data));
    }

    #[bench]
    fn bench_recursive_bisection_inverse_fft(b: &mut test::Bencher) {
        let data = prepare_data();
        b.iter(|| RecursiveBisectionFFT::inverse_fourier_transform(&data));
    }
}
