use numerical::tensor::matrix::{MatrixBaseOps, MatrixSquareFullVec, Square};

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
