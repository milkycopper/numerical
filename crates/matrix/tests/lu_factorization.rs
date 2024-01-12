use floating_point::F64;
use matrix::{FullMat, Matrix};

fn vec_max_diff(a: &Vec<F64>, b: &Vec<F64>) -> F64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x - *y).abs())
        .fold(0.0.into(), |max, x| if max < x { x } else { max })
}

#[test]
fn test_lu() {
    let mat = FullMat::from_vec(3, F64::map_vec(vec![2., 1., 5., 4., 4., -4., 1., 3., 1.]));
    let (l, u, p) = mat.lu().unwrap();
    println!("p = {:?}", p);
    println!("l = \n{}", l);
    println!("u = \n{}", u);

    let mat_b = FullMat::from(l).mul_mat(&FullMat::from(u));
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
fn test_singular_lu() {
    let mat = FullMat::from_vec(
        3,
        F64::map_vec(vec![2., 1., 5., 4., 4., -4., -2., -1., -5.]),
    );
    assert!(mat.lu().is_none())
}

#[test]
fn test_lu_solve_0() {
    let mat = FullMat::from_vec(3, F64::map_vec(vec![2., 1., 5., 4., 4., -4., 1., 3., 1.]));
    let b = F64::map_vec(vec![5., 0., 6.]);
    let x = mat.lu_solve(&b).unwrap();
    assert!(x[0] == F64::from(-1.0));
    assert!(x[1] == F64::from(2.0));
    assert!(x[2] == F64::from(1.0));

    let y = mat.mul_vec(&x);
    assert!(vec_max_diff(&y, &b) == 0.0.into());
}

#[test]
fn test_lu_solve_1() {
    let mat = FullMat::from_vec(3, F64::map_vec(vec![4., 2., 0., 4., 4., -2., 2., 2., 3.]));
    let b = F64::map_vec(vec![2., 4., 6.]);
    let x = mat.lu_solve(&b).unwrap();
    let y = mat.mul_vec(&x);
    assert!(vec_max_diff(&y, &b) == 0.0.into());
}

#[test]
fn test_lu_solve_2() {
    let mat = FullMat::from_vec(3, F64::map_vec(vec![4., 2., 0., 4., 4., -2., 2., 2., 3.]));
    let b = F64::map_vec(vec![2., 4., 6.]);
    let x = mat.lu_solve(&b).unwrap();
    let y = mat.mul_vec(&x);
    assert!(vec_max_diff(&y, &b) < (4.0 * f64::EPSILON).into());
}

#[test]
fn test_inv() {
    let mat = FullMat::from_vec(3, F64::map_vec(vec![4., 2., 0., 4., 4., -2., 2., 2., 3.]));
    let inv_mat = mat.inv().unwrap();
    println!("Inverse mat = {inv_mat}");
    println!("mat * inv_mat = {}", mat.mul_mat(&inv_mat));

    let identity_mat = FullMat::from_vec(3, F64::map_vec(vec![1., 0., 0., 0., 1., 0., 0., 0., 1.]));
    assert!(mat.mul_mat(&inv_mat).sub(&identity_mat).element_max_abs() == 0.0.into());
}

#[test]
fn test_norm_and_condition_number() {
    fn hibert_mat(n: usize) -> FullMat<F64> {
        let mut v = vec![];
        for i in 1..(n + 1) {
            for j in 1..(n + 1) {
                v.push(F64::from(1.0 / (i + j - 1) as f64));
            }
        }

        FullMat::from_vec(n, v)
    }

    fn condition_number(mat: &FullMat<F64>) -> F64 {
        assert!(mat.is_square());

        let inv_mat = mat.inv().unwrap();
        let norm = mat.norm();
        let inv_norm = inv_mat.norm();

        norm * inv_norm
    }

    let n = 6;
    let mat_a = hibert_mat(n);
    let ones = vec![F64::from(1.0); n as usize];
    let b = mat_a.mul_vec(&ones);
    let x = mat_a.lu_solve(&b).unwrap();
    println!("n = {}, x = {:#?}", n, x);
    let conditon_number = condition_number(&mat_a);
    println!(
        "condition number of hibert({}) = {}",
        n,
        format!("{:.6e}", conditon_number.to_f64())
    );
    assert!(conditon_number > 1e7.into());

    let n = 10;
    let mat_a = hibert_mat(n);
    let ones = vec![F64::from(1.0); n as usize];
    let b = mat_a.mul_vec(&ones);
    let x = mat_a.lu_solve(&b).unwrap();
    println!("n = {}, x = {:#?}", n, x);
    let conditon_number = condition_number(&mat_a);
    println!(
        "condition number of hibert({}) = {}",
        n,
        format!("{:.6e}", conditon_number.to_f64())
    );
    assert!(conditon_number > 1e13.into());
}
