use floating_point::F64;

#[test]
fn test_add() {
    assert!(F64::from(3.0) + F64::from(4.0) == 7.0.into())
}

#[test]
fn test_add_assign() {
    let mut x = F64::from(3.0);
    x += F64::from(4.0);
    assert!(x == 7.0.into())
}

#[test]
fn test_sub() {
    assert!(F64::from(3.0) - F64::from(4.0) == -F64::from(1.0))
}

#[test]
fn test_sub_assign() {
    let mut x = F64::from(3.0);
    x -= F64::from(4.0);
    assert!(x == (-1.0).into())
}

#[test]
fn test_mul() {
    assert!(F64::from(3.0) * F64::from(4.0) == 12.0.into())
}

#[test]
fn test_mul_assign() {
    let mut x = F64::from(3.0);
    x *= F64::from(4.0);
    assert!(x == 12.0.into())
}

#[test]
fn test_div() {
    assert!(F64::from(3.0) / F64::from(4.0) == 0.75.into())
}

#[test]
fn test_div_assign() {
    let mut x = F64::from(3.0);
    x /= F64::from(4.0);
    assert!(x == 0.75.into())
}

#[test]
fn test_neg() {
    assert!(-F64::from(3.0) == F64::from(-3.0))
}

#[test]
fn test_to_f64() {
    assert!(F64::from(3.0).to_f64() == 3.0)
}

#[test]
fn test_abs() {
    assert!(F64::from(-3.0).abs() == 3.0.into())
}

#[test]
fn test_map_vec() {
    let v0 = vec![F64::from(1.0); 10];
    let v1 = F64::map_vec(vec![1.0; 10]);
    for i in 0..10 {
        assert!(v0[i] == v1[i])
    }
}

#[test]
fn test_min_max() {
    assert!(F64::from(3.0).max(F64::from(4.0)) == F64::from(4.0));
    assert!(F64::from(3.0).min(F64::from(4.0)) == F64::from(3.0));
}
