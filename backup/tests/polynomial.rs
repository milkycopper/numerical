use numerical::{dim1_func::Dim1Func, polynomial::Polynomial};

#[test]
fn test_nest_mul_0() {
    let poly = Polynomial::with_coefficients(vec![-1., 5., -3., 3., 2.].into())
        .with_base_points(vec![0.; 4].into());
    let y = poly.nest_mul(0.5);
    assert!(y == 1.25);
}

#[test]
fn test_nest_mul_1() {
    let poly = Polynomial::with_coefficients(vec![-1., 5., -3., 3., 2.].into());
    let y = poly.nest_mul(0.5);
    assert!(y == 1.25, "y = {y:?}");
}

#[test]
fn test_nest_mul_2() {
    let poly = Polynomial::with_coefficients(vec![1., 0.5, 0.5, -0.5].into())
        .with_base_points(vec![0., 2., 3.].into());
    let y = poly.nest_mul(1.);
    assert!(y == 0.);
}

#[test]
fn test_eval_0() {
    let poly = Polynomial::with_coefficients(vec![1., 0.5, 0.5, -0.5].into())
        .with_base_points(vec![0., 2., 3.].into());
    let y = poly.eval(1.);
    assert!(y == 0.);
}

#[test]
fn test_eval_1() {
    let poly = Polynomial::with_coefficients(vec![1., 0.5, 0.5, -0.5].into())
        .with_base_points(vec![0., 2., 3.].into());
    let f = |x: f32| 1. + x * (0.5 + (x - 2.) * (0.5 + (x - 3.) * (-0.5)));
    for x in [-99., -9., -0.9, 0.7, 7., 77.] {
        assert!(poly.eval(x) == f(x))
    }
}
