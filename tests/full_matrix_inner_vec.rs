use numerical::tensor::matrix::{MatrixBaseOps, MatrixFullVec};

#[test]
fn test_matrix_ops_0() {
    let inner = vec![1f32, 2., 3., 4., 5., 6., 7., 8., 9.];
    let mut mat = MatrixFullVec::new_with_vec((3, 3).into(), inner);
    assert!(mat.shape() == (3, 3).into());
    assert!(mat.row_size() == 3);
    assert!(mat.col_size() == 3);
    assert!(mat.is_square());
    assert!(mat[(1, 0)] == 4.);
    assert!(mat[(2, 1)] == 8.);
    mat[(1, 0)] = -4.;
    assert!(mat[(1, 0)] == -4.);
}

#[test]
fn test_matrix_ops_1() {
    let inner_0 = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.];
    let inner_1 = vec![2., 3., 4., 5., 6., 7., 8., 9., 10.];
    let inner_2 = vec![1., 2., 3.];
    let inner_3 = vec![35., 40., 45., 77., 91., 105., 119., 142., 165.];

    let mat_0 = MatrixFullVec::new_with_vec((3, 3).into(), inner_0.clone());
    let mat_1 = MatrixFullVec::new_with_vec((3, 3).into(), inner_1.clone());
    let mat_2 = MatrixFullVec::new_with_vec((3, 1).into(), inner_2.clone());
    let mat_3 = MatrixFullVec::new_with_vec((3, 3).into(), inner_3.clone());

    assert!(!mat_2.is_square());

    assert!(mat_0 != mat_1);
    assert!(&mat_0.mul(&mat_1) - &mat_0 == mat_3);
    assert!(&mat_0 * &mat_2 == MatrixFullVec::new_with_vec((3, 1).into(), vec![14., 32., 50.]));

    assert!(
        &mat_0 * &MatrixFullVec::new_with_vec((3, 2).into(), vec![1., 2., 4., 5., 7., 8.])
            == MatrixFullVec::new_with_vec((3, 2).into(), vec![30., 36., 66., 81., 102., 126.])
    );

    assert!(
        &mat_0 + &mat_1
            == MatrixFullVec::new_with_vec(
                (3, 3).into(),
                inner_0.iter().map(|x| 2. * x + 1.).collect()
            )
    );

    assert!(
        mat_0.add(&mat_0)
            == MatrixFullVec::new_with_vec((3, 3).into(), inner_0.iter().map(|x| 2. * x).collect())
    );

    assert!(
        &mat_3 - &mat_0
            == MatrixFullVec::new_with_vec(
                (3, 3).into(),
                vec![34., 38., 42., 73., 86., 99., 112., 134., 156.]
            )
    );

    assert!(
        mat_3.sub(&mat_0)
            == MatrixFullVec::new_with_vec(
                (3, 3).into(),
                vec![34., 38., 42., 73., 86., 99., 112., 134., 156.]
            )
    );
}

#[test]
fn test_matrix_ops_2() {
    let inner_0 = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.];
    let inner_1 = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.];

    let mut mat_0 = MatrixFullVec::new_with_vec((3, 3).into(), inner_0.clone());
    let mat_1 = MatrixFullVec::new_with_vec((3, 3).into(), inner_1.clone());

    mat_0 += &mat_1;
    assert!(
        mat_0
            == MatrixFullVec::new_with_vec((3, 3).into(), inner_0.iter().map(|x| 2. * x).collect())
    );

    mat_0 -= &mat_1;
    assert!(mat_0 == mat_1);

    mat_0.add_assign(&mat_1);
    assert!(
        mat_0
            == MatrixFullVec::new_with_vec((3, 3).into(), inner_0.iter().map(|x| 2. * x).collect())
    );

    mat_0.sub_assign(&mat_1);
    assert!(mat_0 == mat_1);
}

#[test]
fn test_matrix_display() {
    let inner = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.];
    let mat = MatrixFullVec::new_with_vec((3, 3).into(), inner);
    assert!(
        format!("{mat}")
            == "[
[1, 2, 3],
[4, 5, 6],
[7, 8, 9]
]"
    );
}
