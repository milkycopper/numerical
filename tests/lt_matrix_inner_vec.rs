use numerical::tensor::matrix::{MatrixLTVec, Square};

#[test]
fn test_matrix_ops_0() {
    let inner = vec![1., 4., 5., 7., 8., 9.];
    let mut mat = MatrixLTVec::new_with_vec(3, inner);
    assert!(mat.shape() == (3, 3).into());
    assert!(mat.row_size() == 3);
    assert!(mat.col_size() == 3);
    assert!(mat.size() == 3);
    assert!(mat[(1, 0)] == 4.);
    assert!(mat[(2, 1)] == 8.);
    mat[(1, 0)] = -4.;
    assert!(mat[(1, 0)] == -4.);
}

#[test]
fn test_matrix_ops_1() {
    let inner_0 = vec![1f32, 4., 5., 7., 8., 9.];
    let inner_1 = vec![2., 5., 6., 8., 9., 10.];

    let mat_0 = MatrixLTVec::new_with_vec(3, inner_0.clone());
    let mat_1 = MatrixLTVec::new_with_vec(3, inner_1.clone());

    assert!(mat_0 != mat_1);

    assert!(&mat_0 + &mat_1 == MatrixLTVec::new_with_vec(3, vec![3., 9., 11., 15., 17., 19.]));
    assert!(&mat_0 - &mat_1 == MatrixLTVec::new_with_vec(3, vec![-1.; 6]));
    assert!(&mat_0 * &mat_1 == MatrixLTVec::new_with_vec(3, vec![2., 33., 30., 126., 129., 90.]));
}

#[test]
fn test_matrix_ops_2() {
    let inner_0 = vec![1., 4., 5., 7., 8., 9.];
    let inner_1 = vec![2., 5., 6., 8., 9., 10.];

    let mut mat_0 = MatrixLTVec::new_with_vec(3, inner_0.clone());
    let mat_1 = MatrixLTVec::new_with_vec(3, inner_1.clone());

    assert!(mat_0 != mat_1);

    mat_0 += &mat_1;
    assert!(mat_0 == MatrixLTVec::new_with_vec(3, vec![3., 9., 11., 15., 17., 19.]));

    mat_0 -= &mat_1;
    assert!(mat_0 == MatrixLTVec::new_with_vec(3, vec![1., 4., 5., 7., 8., 9.]));
}

#[test]
fn test_matrix_display() {
    let inner = vec![1., 4., 5., 7., 8., 9.];
    let mat = MatrixLTVec::new_with_vec(3, inner);
    assert!(
        format!("{mat}")
            == "[
[1, 0, 0],
[4, 5, 0],
[7, 8, 9]
]"
    );
}
