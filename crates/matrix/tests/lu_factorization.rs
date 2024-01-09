use floating_point::F64;
use matrix::FullMat;

#[test]
fn test_lu() {
    let mat = FullMat::from_vec(
        3,
        vec![2., 1., 5., 4., 4., -4., 1., 3., 1.]
            .into_iter()
            .map(F64::from)
            .collect(),
    );
    let (l, u, p) = mat.lu().unwrap();
    println!("p = {:?}", p);
    println!("l = {}", l);
    println!("u = {}", u);
}

#[test]
fn test_lu_solve_0() {
    let mat = FullMat::from_vec(
        3,
        vec![2., 1., 5., 4., 4., -4., 1., 3., 1.]
            .into_iter()
            .map(F64::from)
            .collect(),
    );
    let y = mat
        .lu_solve(vec![5., 0., 6.].into_iter().map(F64::from).collect())
        .unwrap();
    assert!(y[0] == F64::from(-1.0));
    assert!(y[1] == F64::from(2.0));
    assert!(y[2] == F64::from(1.0));
}
