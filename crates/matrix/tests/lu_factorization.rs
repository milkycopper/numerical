use floating_point::F64;
use matrix::FullMat;

#[test]
fn test_lu() {
    let mat = FullMat::from_rows(vec![
        vec![2., 1., 5.].into_iter().map(F64::from).collect(),
        vec![4., 4., -4.].into_iter().map(F64::from).collect(),
        vec![1., 3., 1.].into_iter().map(F64::from).collect(),
    ]);
    let (p, l, u) = mat.lu();
    println!("p = {:?}", p);
    println!("l = {:?}", l);
    println!("u = {:?}", u);
}
