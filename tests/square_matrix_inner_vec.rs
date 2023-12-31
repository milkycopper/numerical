use approx::AbsDiffEq;
use numerical::tensor::{
    matrix::{
        LUFactorization, MatrixAddSubSelf, MatrixLTVec, MatrixMulSelf, MatrixPermutationVec,
        MatrixSquareFullVec, MatrixUTVec, Square,
    },
    vector::Vector,
};

#[test]
fn test_matrix_ops_0() {
    let inner = vec![1f32, 2., 3., 4., 5., 6., 7., 8., 9.];
    let mut mat = MatrixSquareFullVec::new_with_vec(3, inner);
    assert!(mat.shape() == (3, 3).into());
    assert!(mat.row_size() == 3);
    assert!(mat.col_size() == 3);
    assert!(mat.is_square());
    assert!(mat.size() == 3);
    assert!(mat[(1, 0)] == 4.);
    assert!(mat[(2, 1)] == 8.);
    mat[(1, 0)] = -4.;
    assert!(mat[(1, 0)] == -4.);
}

#[test]
fn test_matrix_ops_1() {
    let inner_0 = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.];
    let inner_1 = vec![2., 3., 4., 5., 6., 7., 8., 9., 10.];
    let inner_3 = vec![35., 40., 45., 77., 91., 105., 119., 142., 165.];

    let mat_0 = MatrixSquareFullVec::new_with_vec(3, inner_0.clone());
    let mat_1 = MatrixSquareFullVec::new_with_vec(3, inner_1.clone());
    let mat_3 = MatrixSquareFullVec::new_with_vec(3, inner_3.clone());

    assert!(mat_0 != mat_1);
    assert!(&mat_0.mul(&mat_1) - &mat_0 == mat_3);

    assert!(
        &mat_0 + &mat_1
            == MatrixSquareFullVec::new_with_vec(3, inner_0.iter().map(|x| 2. * x + 1.).collect())
    );

    assert!(
        mat_0.add(&mat_0)
            == MatrixSquareFullVec::new_with_vec(3, inner_0.iter().map(|x| 2. * x).collect())
    );

    assert!(
        &mat_3 - &mat_0
            == MatrixSquareFullVec::new_with_vec(
                3,
                vec![34., 38., 42., 73., 86., 99., 112., 134., 156.]
            )
    );

    assert!(
        mat_3.sub(&mat_0)
            == MatrixSquareFullVec::new_with_vec(
                3,
                vec![34., 38., 42., 73., 86., 99., 112., 134., 156.]
            )
    );
}

#[test]
fn test_matrix_ops_2() {
    let inner_0 = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.];
    let inner_1 = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.];

    let mut mat_0 = MatrixSquareFullVec::new_with_vec(3, inner_0.clone());
    let mat_1 = MatrixSquareFullVec::new_with_vec(3, inner_1.clone());

    mat_0 += &mat_1;
    assert!(
        mat_0 == MatrixSquareFullVec::new_with_vec(3, inner_0.iter().map(|x| 2. * x).collect())
    );

    mat_0 -= &mat_1;
    assert!(mat_0 == mat_1);

    mat_0.add_assign(&mat_1);
    assert!(
        mat_0 == MatrixSquareFullVec::new_with_vec(3, inner_0.iter().map(|x| 2. * x).collect())
    );

    mat_0.sub_assign(&mat_1);
    assert!(mat_0 == mat_1);
}

#[test]
fn test_matrix_display() {
    let inner = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.];
    let mat = MatrixSquareFullVec::new_with_vec(3, inner);
    assert!(
        format!("{mat}")
            == "[
[1, 2, 3],
[4, 5, 6],
[7, 8, 9]
]"
    );
}

#[test]
fn test_matrix_transpose() {
    let inner = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.];
    let mat = MatrixSquareFullVec::new_with_vec(3, inner);
    assert!(
        format!("{}", mat.transpose())
            == "[
[1, 4, 7],
[2, 5, 8],
[3, 6, 9]
]"
    );
}

#[test]
fn test_lu_factorization_0() {
    let mat = MatrixSquareFullVec::new_with_vec(3, vec![1., 2., -1., 2., 1., -2., -3., 1., 1.]);
    let (lt, ut) = mat.lu();
    assert!(lt.abs_diff_eq(
        &MatrixLTVec::new_with_vec(3, vec![1., 2., 1., -3., -7. / 3., 1.]),
        f64::EPSILON * 2.
    ));
    assert!(ut.abs_diff_eq(
        &MatrixUTVec::new_with_vec(3, vec![1., 2., -1., -3., 0., -2.]),
        f64::EPSILON * 2.
    ));
    let mul_res = MatrixSquareFullVec::from(lt) * &MatrixSquareFullVec::from(ut);
    assert!(mul_res.abs_diff_eq(&mat, f64::EPSILON * 4.));

    let mat = MatrixSquareFullVec::new_with_vec(
        4,
        vec![
            1., -1., 1., 2., 0., 2., 1., 0., 1., 3., 4., 4., 0., 2., 1., -1.,
        ],
    );
    let (lt, ut) = mat.lu();
    let mul_res = MatrixSquareFullVec::from(lt) * &MatrixSquareFullVec::from(ut);
    assert!(mul_res.abs_diff_eq(&mat, f64::EPSILON * 4.));

    let mat = MatrixSquareFullVec::new_with_vec(
        5,
        vec![
            1., -1., 1., 2., 8.2, 0.99, 2., 1., 0.134, 2.9, 1.65, 3.34, 4.93, 4., 3.7, 0.24, 2.,
            1., -1., 4.4, 1., 2., 3., 4., 5.,
        ],
    );
    let (lt, ut) = mat.lu();
    let mul_res = MatrixSquareFullVec::from(lt) * &MatrixSquareFullVec::from(ut);
    assert!(mul_res.abs_diff_eq(&mat, f64::EPSILON * 4.));

    let mat = MatrixSquareFullVec::new_with_vec(
        5,
        vec![
            1f32, -1., 1., 2., 8.2, 0.99, 2., 1., 0.134, 2.9, 1.65, 3.34, 4.93, 4., 3.7, 0.24, 2.,
            1., -1., 4.4, 1., 2., 3., 4., 5.,
        ],
    );
    let (lt, ut) = mat.lu();
    let mul_res = MatrixSquareFullVec::from(lt) * &MatrixSquareFullVec::from(ut);
    assert!(mul_res.abs_diff_eq(&mat, f32::EPSILON * 4.));
}

#[test]
fn test_lu_factorization_1() {
    let mat = MatrixSquareFullVec::new_with_vec(3, vec![3., 1., 2., 6., 1., 4., 3., 1., 5.]);
    let (lt, ut) = mat.lu();
    let mul_res = MatrixSquareFullVec::from(lt) * &MatrixSquareFullVec::from(ut);
    assert!(mul_res.abs_diff_eq(&mat, f32::EPSILON));
}

#[test]
fn test_lu_factorization_2() {
    let mat = MatrixSquareFullVec::new_with_vec(3, vec![4., 2., 0., 4., 4., 2., 2., 2., 3.]);
    let (lt, ut) = mat.lu();
    let mul_res = MatrixSquareFullVec::from(lt) * &MatrixSquareFullVec::from(ut);
    assert!(mul_res.abs_diff_eq(&mat, f32::EPSILON));
}

#[test]
fn test_lu_factorization_3() {
    let mat = MatrixSquareFullVec::new_with_vec(
        4,
        vec![
            1., -1., 1., 2., 0., 2., 1., 0., 1., 3., 4., 4., 0., 2., 1., -1.,
        ],
    );
    let (lt, ut) = mat.lu();
    let mul_res = MatrixSquareFullVec::from(lt) * &MatrixSquareFullVec::from(ut);
    assert!(mul_res.abs_diff_eq(&mat, f32::EPSILON));
}

#[test]
fn test_lu_solve_0() {
    let mat_a = MatrixSquareFullVec::new_with_vec(3, vec![4.8, 2., 0., 4.3, 4., 2., 2., 2., 3.]);
    let b = Vector::new(vec![2., 4., 6.]);
    let x = mat_a.lu_solve(&b);
    println!("solution x = {x:#?}");
    let y = mat_a.transform_vector(&x);
    assert!(b.abs_diff_eq(&y, f64::EPSILON * 2.))
}

#[test]
#[should_panic(expected = "pivot element at")]
fn test_lu_solve_1() {
    let mat_lt = MatrixLTVec::new_with_vec(3, vec![0., 1., 3., 4., 1., 2.])
        .extend_with_diagonal(&mut (0..4).map(|_| 1.0));
    let mat_ut = MatrixLTVec::new_with_vec(4, vec![2., 1., 0., 0., 1., 2., 0., -1., 1., 1.]);
    let mat = MatrixSquareFullVec::from(mat_lt) * &MatrixSquareFullVec::from(mat_ut);
    println!("mat x = {mat}");
    let b = Vector::new(vec![1., 1., 2., 0.]);
    let x = mat.lu_solve(&b);
    println!("solution x = {x:#?}");
    let y = mat.transform_vector(&x);
    assert!(b.abs_diff_eq(&y, f64::EPSILON * 2.))
}

#[test]
#[should_panic]
fn test_lu_solve_2() {
    let mat_a = MatrixSquareFullVec::new_with_vec(2, vec![1e-20, 1., 1., 2.]);
    let b = Vector::new(vec![1., 4.]);
    let x = mat_a.lu_solve(&b);
    println!("solution x = {x:#?}");
    let y = mat_a.transform_vector(&x);
    println!("y = {x:#?}");
    assert!(b.abs_diff_eq(&y, f64::EPSILON * 2.))
}

#[test]
fn test_matrix_exchange_row() {
    let inner = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.];
    let mut mat = MatrixSquareFullVec::new_with_vec(3, inner);
    mat.exchange_row(0, 1);
    assert!(
        format!("{mat}")
            == "[
[4, 5, 6],
[1, 2, 3],
[7, 8, 9]
]"
    );
}

#[test]
fn test_pa_lu_0() {
    let mat = MatrixSquareFullVec::new_with_vec(
        3,
        vec![2, 1, 5, 4, 4, -4, 1, 3, 1]
            .iter()
            .map(|x| *x as f64)
            .collect(),
    );

    let (p, l, u) = mat.pa_lu();
    assert!(p == MatrixPermutationVec::new_from_vec(vec![1, 2, 0]));
    assert!(l.abs_diff_eq(
        &MatrixLTVec::new_with_vec(3, vec![1., 0.25, 1., 0.5, -0.5, 1.]),
        f64::EPSILON
    ));
    assert!(u.abs_diff_eq(
        &MatrixUTVec::new_with_vec(
            3,
            vec![4, 4, -4, 2, 2, 8].iter().map(|x| *x as f64).collect()
        ),
        f64::EPSILON
    ));
}

#[test]
fn test_pa_lu_solve_0() {
    let mat_a = MatrixSquareFullVec::new_with_vec(3, vec![4.8, 2., 0., 4.3, 4., 2., 2., 2., 3.]);
    let b = Vector::new(vec![2., 4., 6.]);
    let x = mat_a.pa_lu_solve(&b);
    println!("solution x = {x:#?}");
    let y = mat_a.transform_vector(&x);
    assert!(b.abs_diff_eq(&y, f64::EPSILON * 2.))
}

#[test]
fn test_pa_lu_solve_1() {
    let mat_a =
        MatrixSquareFullVec::new_with_vec(10, (0..100).map(|x| x as f64 + 0.3f64).collect());
    let b = Vector::new((0..10).map(|x| x as f64 - 6.3f64).collect::<Vec<f64>>());
    let x = mat_a.pa_lu_solve(&b);
    println!("solution x = {x:#?}");
    let y = mat_a.transform_vector(&x);
    println!("mat_a * x = {y:#?}");
    println!("b = {b:#?}");
    assert!(b.abs_diff_eq(&y, f64::EPSILON * 256.))
}

#[test]
fn test_pa_lu_solve_2() {
    let mat_a = MatrixSquareFullVec::new_with_vec(2, vec![1e-20, 1., 1., 2.]);
    let b = Vector::new(vec![1., 4.]);
    let x = mat_a.pa_lu_solve(&b);
    println!("solution x = {x:#?}");
    let y = mat_a.transform_vector(&x);
    println!("y = {x:#?}");
    assert!(b.abs_diff_eq(&y, f64::EPSILON))
}
