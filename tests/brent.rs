use numerical::{
    base_float::BaseFloat,
    dim1_func::Dim1Func,
    dim1_solver::{brent::BrentSolver, Dim1Solver},
    polynomial::PolynomialInnerVec,
};

#[test]
fn test_brent_0() {
    let _ = env_logger::try_init();
    let f = PolynomialInnerVec::from_coefficients(vec![-1., 1., 0., 1.].into());
    let solver = BrentSolver::new([0., 1.], 1e-14);
    let result = solver.solve(&f).unwrap();
    // large forward error
    assert!((result - 0.682328).abs() < 1e-6, "root = {}", result);
}

#[test]
fn test_brent_1() {
    let _ = env_logger::try_init();
    let f =
        PolynomialInnerVec::from_coefficients(vec![-4., 16., 35., -69., -102., 45., 54.].into());
    let ranges = [[-1.5, -1.], [0., 0.3], [0.3, 0.8], [0.8, 1.5]];
    let solvers = ranges.map(|r| BrentSolver::new(r, 1e-14));
    let roots = solvers.map(|s| s.solve(&f).unwrap());
    let backward_errors = roots.map(|x| f.eval(x));
    println!("roots = {:?}", roots);
    for i in 0..roots.len() {
        assert!(backward_errors[i].abs() < 1e-8,);
    }
}

#[test]
fn test_brent_2() {
    let _ = env_logger::try_init();
    let f = PolynomialInnerVec::<f32>::from_coefficients(
        vec![-4., 16., 35., -69., -102., 45., 54.].into(),
    );
    let ranges = [[-1.5, -1.], [0., 0.3], [0.3, 0.8], [0.8, 1.5]];
    let solvers = ranges.map(|r| BrentSolver::new(r, 1e-6));
    let roots = solvers.map(|s| s.solve(&f).unwrap());
    let backward_errors = roots.map(|x| f.eval(x));
    println!("roots = {:?}", roots);
    for i in 0..roots.len() {
        assert!(backward_errors[i].abs() < 1e-3,);
    }
}

#[test]
fn test_brent_illegal_range() {
    let _ = env_logger::try_init();
    let f = PolynomialInnerVec::<f32>::from_coefficients(vec![1., -2., 1.].into());
    let solver = BrentSolver::new([0., 2.], 1e-6);
    let result = std::panic::catch_unwind(|| solver.solve(&f));
    assert!(result.is_err());
}
