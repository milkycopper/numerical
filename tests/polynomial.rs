use numerical::{
    dim1_func::Dim1Func,
    polynomial::{Polynomial, PolynomialInnerVec},
    tensor::vector::Vector,
};

#[test]
fn test_nest_mul_inner_array() {
    let poly = Polynomial::new(
        Vector::<f64, [f64; 5]>::new([-1., 5., -3., 3., 2.]),
        Some(Vector::<f64, [f64; 4]>::new([0.; 4])),
    );
    let y = poly.nest_mul(0.5);
    assert!(y == 1.25);
}

#[test]
fn test_nest_mul_inner_array_and_vec() {
    let poly = Polynomial::new(
        Vector::<f64, [f64; 5]>::new([-1., 5., -3., 3., 2.]),
        Some(Vector::<f64, Vec<f64>>::new(vec![0., 0., 0., 0.])),
    );
    let y = poly.nest_mul(0.5);
    assert!(y == 1.25);
}

#[test]
fn test_nest_mul_inner_vec_0() {
    let poly = PolynomialInnerVec::new(
        vec![-1., 5., -3., 3., 2.].into(),
        Some(vec![0., 0., 0., 0.].into()),
    );
    let y = poly.nest_mul(0.5);
    assert!(y == 1.25);
}

#[test]
fn test_new_0() {
    let poly = PolynomialInnerVec::from_coefficients(vec![-1., 5., -3., 3., 2.].into());
    let y = poly.nest_mul(0.5);
    assert!(y == 1.25, "y = {y:?}");
}

#[test]
fn test_nest_mul_inner_vec_1() {
    let poly = PolynomialInnerVec::new(
        vec![1., 0.5, 0.5, -0.5].into(),
        Some(vec![0., 2., 3.].into()),
    );
    let y = poly.nest_mul(1.);
    assert!(y == 0.);
}

#[test]
fn test_new_1() {
    let poly = PolynomialInnerVec::from_coefficients_base_points(
        vec![1., 0.5, 0.5, -0.5].into(),
        vec![0., 2., 3.].into(),
    );
    let y = poly.nest_mul(1.);
    assert!(y == 0.);
}

#[test]
fn test_eval_0() {
    let poly = PolynomialInnerVec::from_coefficients_base_points(
        vec![1., 0.5, 0.5, -0.5].into(),
        vec![0., 2., 3.].into(),
    );
    let y = poly.eval(1.);
    assert!(y == 0.);
}

#[test]
fn test_eval_1() {
    let poly = PolynomialInnerVec::<f32>::from_coefficients_base_points(
        vec![1., 0.5, 0.5, -0.5].into(),
        vec![0., 2., 3.].into(),
    );
    let f = |x: f32| 1. + x * (0.5 + (x - 2.) * (0.5 + (x - 3.) * (-0.5)));
    for x in [-99., -9., -0.9, 0.7, 7., 77.] {
        assert!(poly.eval(x) == f(x))
    }
}
