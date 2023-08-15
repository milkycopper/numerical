use numerical::tensor::{
    matrix::{MatrixPermutationVec, MatrixSquareFullVec},
    vector::VectorInnerVec,
};

#[test]
fn test_exchange() {
    let mut pm = MatrixPermutationVec::<f64>::identity(10);
    for i in 0..10 {
        for j in 0..10 {
            if i == j {
                assert!(pm[(i, j)] == 1f64);
            } else {
                assert!(pm[(i, j)] == 0f64);
            }
        }
    }

    pm.exchange(3, 6);

    assert!(pm[(3, 6)] == 1f64);
    assert!(pm[(3, 3)] == 0f64);
    assert!(pm[(6, 3)] == 1f64);
    assert!(pm[(6, 6)] == 0f64);

    println!("pm = {pm}");
}

#[test]
fn test_mul_self() {
    let pm_0 = MatrixPermutationVec::<f64>::new_from_vec(vec![0, 1, 3, 2]);
    let pm_1 = MatrixPermutationVec::<f64>::new_from_vec(vec![1, 0, 2, 3]);
    let pm_2 = pm_0 * &pm_1;
    assert!(
        format!("{pm_2}")
            == "[
[0, 1, 0, 0],
[1, 0, 0, 0],
[0, 0, 0, 1],
[0, 0, 1, 0]
]"
    );
}

#[test]
fn test_mul_square() {
    let pm_0 = MatrixPermutationVec::<f64>::new_from_vec(vec![0, 1, 3, 2]);
    let pm_1 = MatrixPermutationVec::<f64>::new_from_vec(vec![1, 0, 2, 3]);
    let pm_2 = pm_0 * &pm_1;
    let m = pm_2 * &MatrixSquareFullVec::new_with_vec(4, (0..16).map(|x| x as f64).collect());
    assert!(
        format!("{m}")
            == "[
[4, 5, 6, 7],
[0, 1, 2, 3],
[12, 13, 14, 15],
[8, 9, 10, 11]
]"
    );
}

#[test]
fn test_mul_vector() {
    let pm_0 = MatrixPermutationVec::<f64>::new_from_vec(vec![0, 1, 3, 2]);
    let pm_1 = MatrixPermutationVec::<f64>::new_from_vec(vec![1, 0, 2, 3]);
    let pm_2 = pm_0 * &pm_1;
    let v = pm_2 * &VectorInnerVec::new(vec![0., 1., 2., 3.]);
    assert!(v[0] == 1.0);
    assert!(v[1] == 0.0);
    assert!(v[2] == 3.0);
    assert!(v[3] == 2.0);
}

#[test]
fn test_mul_self_1() {
    let pm_0 = MatrixPermutationVec::<f64>::new_from_vec(vec![0, 1, 3, 2]);
    let pm_1 = MatrixPermutationVec::<f64>::new_from_vec(vec![1, 0, 2, 3]);
    let pm_2 = &pm_0 * &pm_1;
    assert!(
        format!("{pm_2}")
            == "[
[0, 1, 0, 0],
[1, 0, 0, 0],
[0, 0, 0, 1],
[0, 0, 1, 0]
]"
    );
}

#[test]
fn test_mul_square_1() {
    let pm_0 = MatrixPermutationVec::<f64>::new_from_vec(vec![0, 1, 3, 2]);
    let pm_1 = MatrixPermutationVec::<f64>::new_from_vec(vec![1, 0, 2, 3]);
    let pm_2 = pm_0 * &pm_1;
    let m = &pm_2 * &MatrixSquareFullVec::new_with_vec(4, (0..16).map(|x| x as f64).collect());
    assert!(
        format!("{m}")
            == "[
[4, 5, 6, 7],
[0, 1, 2, 3],
[12, 13, 14, 15],
[8, 9, 10, 11]
]"
    );
}

#[test]
fn test_mul_vector_1() {
    let pm_0 = MatrixPermutationVec::<f64>::new_from_vec(vec![0, 1, 3, 2]);
    let pm_1 = MatrixPermutationVec::<f64>::new_from_vec(vec![1, 0, 2, 3]);
    let pm_2 = pm_0 * &pm_1;
    let v = &pm_2 * &VectorInnerVec::new(vec![0., 1., 2., 3.]);
    assert!(v[0] == 1.0);
    assert!(v[1] == 0.0);
    assert!(v[2] == 3.0);
    assert!(v[3] == 2.0);
}
