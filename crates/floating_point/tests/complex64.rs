#[cfg(feature = "approx")]
use approx::AbsDiffEq;
use floating_point::Complex64;

#[test]
fn test_consts() {
    assert_eq!(Complex64::ZERO, Complex64::new(0., 0.));
    assert_eq!(Complex64::ONE, Complex64::new(1., 0.));
    assert_eq!(Complex64::I, Complex64::new(0., 1.));
}

#[test]
fn test_conjugate() {
    assert_eq!(
        Complex64::new(0.1, 0.2).conjugate(),
        Complex64::from((0.1, -0.2))
    );
    let a = Complex64::new(0.1, 0.2);
    assert_eq!(a.conjugate().conjugate(), a);
}

#[test]
#[cfg(feature = "approx")]
fn test_add() {
    let mut a = Complex64::new(0.1, 0.2);
    let b = Complex64::new(0.3, 0.4);
    let c = Complex64::new(0.4, 0.6);
    assert!(
        (a + b).abs_diff_eq(&c, f64::EPSILON.into()),
        "a + b = {}, c = {}",
        a + b,
        c
    );
    a += b;
    assert!(a.abs_diff_eq(&c, f64::EPSILON.into()));
}

#[test]
#[cfg(feature = "approx")]
fn test_sub() {
    let mut a = Complex64::new(0.1, 0.2);
    let b = Complex64::new(0.3, 0.4);
    let c = Complex64::new(-0.2, -0.2);
    assert!((a - b).abs_diff_eq(&c, f64::EPSILON.into()));
    a -= b;
    assert!(a.abs_diff_eq(&c, f64::EPSILON.into()));

    let d = b + c;
    let f = d - c;
    assert!(b.abs_diff_eq(&f, f64::EPSILON.into()));
}

#[test]
#[cfg(feature = "approx")]
fn test_mul() {
    let mut a = Complex64::new(0.1, 0.2);
    let b = Complex64::new(0.3, 0.4);
    let c = Complex64::new(-0.05, 0.1);
    assert!((a * b).abs_diff_eq(&c, f64::EPSILON.into()));
    a *= b;
    assert!(a.abs_diff_eq(&c, f64::EPSILON.into()));
}

#[test]
#[cfg(feature = "approx")]
fn test_div() {
    let mut a = Complex64::new(0.1, 0.2);
    let b = Complex64::new(0.3, 0.4);
    let c = Complex64::new(0.44, 0.08);
    assert!((a / b).abs_diff_eq(&c, f64::EPSILON.into()));
    a /= b;
    assert!(
        a.abs_diff_eq(&c, f64::EPSILON.into()),
        "a = {}, c = {}",
        a,
        c
    );

    let d = b / c;
    let f = d * c;
    assert!(b.abs_diff_eq(&f, f64::EPSILON.into()));
}

#[test]
fn test_neg() {
    assert_eq!(-Complex64::new(0.3, -0.4), Complex64::new(-0.3, 0.4));

    let a = Complex64::new(0.3, -0.4);
    assert_eq!(-(-a), a);
}
