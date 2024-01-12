use floating_point::F64;
use matrix::{TriFullMat, TriangleMatType};

#[test]
fn test_index() {
    let upper_tri_mat_0 = TriFullMat::from_rows(
        TriangleMatType::Upper,
        vec![
            F64::map_vec(vec![1., 2., 3.]),
            F64::map_vec(vec![4., 5.]),
            F64::map_vec(vec![6.]),
        ],
    );
    let lower_tri_mat_0 = TriFullMat::from_rows(
        TriangleMatType::Lower,
        vec![
            F64::map_vec(vec![1.]),
            F64::map_vec(vec![2., 3.]),
            F64::map_vec(vec![4., 5., 6.]),
        ],
    );

    let upper_tri_mat_1 = TriFullMat::from_vec(
        TriangleMatType::Upper,
        F64::map_vec(vec![1., 2., 3., 4., 5., 6.]),
    );
    let lower_tri_mat_1 = TriFullMat::from_vec(
        TriangleMatType::Lower,
        F64::map_vec(vec![1., 2., 3., 4., 5., 6.]),
    );

    println!("upper triangle matrix = \n{}", upper_tri_mat_0);
    println!("lower triangle matrix = \n{}", lower_tri_mat_0);

    assert!(lower_tri_mat_0[(0, 0)] == 1.0.into());
    assert!(lower_tri_mat_0[(0, 1)] == 0.0.into());
    assert!(lower_tri_mat_0[(0, 2)] == 0.0.into());
    assert!(lower_tri_mat_0[(1, 0)] == 2.0.into());
    assert!(lower_tri_mat_0[(1, 1)] == 3.0.into());
    assert!(lower_tri_mat_0[(1, 2)] == 0.0.into());
    assert!(lower_tri_mat_0[(2, 0)] == 4.0.into());
    assert!(lower_tri_mat_0[(2, 1)] == 5.0.into());
    assert!(lower_tri_mat_0[(2, 2)] == 6.0.into());

    assert!(lower_tri_mat_1[(0, 0)] == 1.0.into());
    assert!(lower_tri_mat_1[(0, 1)] == 0.0.into());
    assert!(lower_tri_mat_1[(0, 2)] == 0.0.into());
    assert!(lower_tri_mat_1[(1, 0)] == 2.0.into());
    assert!(lower_tri_mat_1[(1, 1)] == 3.0.into());
    assert!(lower_tri_mat_1[(1, 2)] == 0.0.into());
    assert!(lower_tri_mat_1[(2, 0)] == 4.0.into());
    assert!(lower_tri_mat_1[(2, 1)] == 5.0.into());
    assert!(lower_tri_mat_1[(2, 2)] == 6.0.into());

    assert!(upper_tri_mat_0[(0, 0)] == 1.0.into());
    assert!(upper_tri_mat_0[(0, 1)] == 2.0.into());
    assert!(upper_tri_mat_0[(0, 2)] == 3.0.into());
    assert!(upper_tri_mat_0[(1, 0)] == 0.0.into());
    assert!(upper_tri_mat_0[(1, 1)] == 4.0.into());
    assert!(upper_tri_mat_0[(1, 2)] == 5.0.into());
    assert!(upper_tri_mat_0[(2, 0)] == 0.0.into());
    assert!(upper_tri_mat_0[(2, 1)] == 0.0.into());
    assert!(upper_tri_mat_0[(2, 2)] == 6.0.into());

    assert!(upper_tri_mat_1[(0, 0)] == 1.0.into());
    assert!(upper_tri_mat_1[(0, 1)] == 2.0.into());
    assert!(upper_tri_mat_1[(0, 2)] == 3.0.into());
    assert!(upper_tri_mat_1[(1, 0)] == 0.0.into());
    assert!(upper_tri_mat_1[(1, 1)] == 4.0.into());
    assert!(upper_tri_mat_1[(1, 2)] == 5.0.into());
    assert!(upper_tri_mat_1[(2, 0)] == 0.0.into());
    assert!(upper_tri_mat_1[(2, 1)] == 0.0.into());
    assert!(upper_tri_mat_1[(2, 2)] == 6.0.into());
}
