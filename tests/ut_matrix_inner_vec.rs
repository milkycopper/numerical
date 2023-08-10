use numerical::tensor::matrix::{MatrixUTVec, Square};

#[test]
fn test_matrix_ops_0() {
    let inner = vec![1., 2., 3., 5., 6., 9.];
    let mut mat = MatrixUTVec::new_with_vec(3, inner);
    assert!(mat.shape() == (3, 3).into());
    assert!(mat.row_size() == 3);
    assert!(mat.col_size() == 3);
    assert!(mat.size() == 3);
    assert!(mat[(1, 2)] == 6.);
    assert!(mat[(2, 2)] == 9.);
    mat[(1, 2)] = -6.;
    assert!(mat[(1, 2)] == -6.);
}

#[test]
fn test_matrix_ops_1() {
    let inner_0 = vec![1., 4., 5., 7., 8., 9.];
    let inner_1 = vec![2., 5., 6., 8., 9., 10.];

    let mat_0 = MatrixUTVec::new_with_vec(3, inner_0.clone());
    let mat_1 = MatrixUTVec::new_with_vec(3, inner_1.clone());

    assert!(mat_0 != mat_1);

    assert!(&mat_0 + &mat_1 == MatrixUTVec::new_with_vec(3, vec![3., 9., 11., 15., 17., 19.]));
    assert!(&mat_0 - &mat_1 == MatrixUTVec::new_with_vec(3, vec![-1.; 6]));
    assert!(&mat_0 * &mat_1 == MatrixUTVec::new_with_vec(3, vec![2., 37., 92., 56., 143., 90.]));
}

#[test]
fn test_matrix_ops_2() {
    let inner_0 = vec![1f32, 4., 5., 7., 8., 9.];
    let inner_1 = vec![2., 5., 6., 8., 9., 10.];

    let mut mat_0 = MatrixUTVec::new_with_vec(3, inner_0.clone());
    let mat_1 = MatrixUTVec::new_with_vec(3, inner_1.clone());

    assert!(mat_0 != mat_1);

    mat_0 += &mat_1;
    assert!(mat_0 == MatrixUTVec::new_with_vec(3, vec![3., 9., 11., 15., 17., 19.]));

    mat_0 -= &mat_1;
    assert!(mat_0 == MatrixUTVec::new_with_vec(3, vec![1., 4., 5., 7., 8., 9.]));
}

#[test]
fn test_matrix_display() {
    let inner = vec![1., 4., 5., 7., 8., 9.];
    let mat = MatrixUTVec::new_with_vec(3, inner);
    assert!(
        format!("{mat}")
            == "[
[1, 4, 5],
[0, 7, 8],
[0, 0, 9]
]"
    );
}
