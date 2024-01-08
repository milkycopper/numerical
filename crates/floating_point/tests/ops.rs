use floating_point::F64;

#[test]
fn test_add() {
    assert!(F64::from(3.0) + F64::from(4.0) == 7.0.into())
}

#[test]
fn test_sub() {
    assert!(F64::from(3.0) - F64::from(4.0) == -F64::from(1.0))
}

#[test]
fn test_mul() {
    assert!(F64::from(3.0) * F64::from(4.0) == 12.0.into())
}

#[test]
fn test_div() {
    assert!(F64::from(3.0) / F64::from(4.0) == 0.75.into())
}
