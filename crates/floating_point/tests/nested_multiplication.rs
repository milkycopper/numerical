use floating_point::F64;
use std::slice::Iter;

mod nested_mul {
    use super::*;

    pub fn nested_mul(x: F64, coes: Iter<F64>) -> F64 {
        let mut ret: F64 = 0.0.into();
        for c in coes {
            ret = ret * x + *c;
        }
        ret
    }
}

#[test]
fn test_simple_nested_mul() {
    let x = F64::from(2.0);
    let coes = [1., 2., 3.].map(F64::from);
    assert!(nested_mul::nested_mul(x, coes.iter()) == 11.0.into())
}

#[test]
fn test_round_off_error() {
    fn nested_mul_x(x: F64) -> F64 {
        let coes = [F64::from(1.0); 51];
        nested_mul::nested_mul(x, coes.iter())
    }

    fn geometric_series_x(x: F64) -> F64 {
        let x1 = x;
        let x2 = x * x;
        let x4 = x2 * x2;
        let x8 = x4 * x4;
        let x16 = x8 * x8;
        let x32 = x16 * x16;
        let x51 = x32 * x16 * x2 * x;
        (x51 - 1.0.into()) / (x1 - 1.0.into())
    }

    let y_a = nested_mul_x(1.00001.into());
    let y_b = geometric_series_x(1.00001.into());

    println!("nested mul result = {:?}", y_a);
    println!("geometric series result = {:?}", y_b);
    let relative_error = ((y_a - y_b) / y_a).abs();
    println!(
        "relative error = {}",
        format!("{:.2e}", relative_error.to_f64())
    );
    println!(
        "machine epsilon = {}",
        format!("{:.2e}", 1.0 / 2_f64.powi(52))
    );
    assert!(relative_error > 1e-12.into())
}
