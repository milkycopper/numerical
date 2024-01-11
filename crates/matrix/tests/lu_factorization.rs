use floating_point::F64;
use matrix::FullMat;

fn vec_max_diff(a: &Vec<F64>, b: &Vec<F64>) -> F64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x - *y).abs())
        .fold(0.0.into(), |max, x| if max < x { x } else { max })
}

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

    let mat_b = l.mul_mat(&u);
    let mat_p = FullMat::from_rows(
        p.iter()
            .map(|i| {
                let mut v = vec![F64::from(0.0); p.len()];
                v[*i as usize] = 1.0.into();
                v
            })
            .collect(),
    );
    let mat_pa = mat_p.mul_mat(&mat);
    let mat_diff = mat_b.sub(&mat_pa);
    let max_diff_abs = mat_diff.element_max_abs();

    assert!(max_diff_abs == 0.0.into());
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
    let b = vec![5., 0., 6.].into_iter().map(F64::from).collect();
    let x = mat.lu_solve(&b).unwrap();
    assert!(x[0] == F64::from(-1.0));
    assert!(x[1] == F64::from(2.0));
    assert!(x[2] == F64::from(1.0));

    let y = mat.mul_vec(&x);
    assert!(vec_max_diff(&y, &b) == 0.0.into());
}

#[test]
fn test_lu_solve_1() {
    let mat = FullMat::from_vec(
        3,
        vec![4., 2., 0., 4., 4., -2., 2., 2., 3.]
            .into_iter()
            .map(F64::from)
            .collect(),
    );
    let b = vec![2., 4., 6.].into_iter().map(F64::from).collect();
    let x = mat.lu_solve(&b).unwrap();
    let y = mat.mul_vec(&x);
    assert!(vec_max_diff(&y, &b) == 0.0.into());
}

#[test]
fn test_lu_solve_2() {
    let mat = FullMat::from_vec(
        3,
        vec![4., 2., 0., 4., 4., -2., 2., 2., 3.]
            .into_iter()
            .map(|x| F64::from(x + 0.1))
            .collect(),
    );
    let b = vec![2., 4., 6.].into_iter().map(F64::from).collect();
    let x = mat.lu_solve(&b).unwrap();
    let y = mat.mul_vec(&x);
    assert!(vec_max_diff(&y, &b) < (4.0 * f64::EPSILON).into());
}
