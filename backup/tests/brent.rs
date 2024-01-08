use numerical::{
    base_float::BaseFloat,
    dim1_func::Dim1Func,
    dim1_solver::{BrentSolver, Dim1Solver},
    polynomial::Polynomial,
};

#[test]
fn test_brent_0() {
    let _ = env_logger::try_init();
    let f = Polynomial::with_coefficients(vec![-1., 1., 0., 1.].into());
    let solver = BrentSolver::with_search_range([0., 1.]).with_error_tolerance(1e-14);
    let result = solver.solve(&f).unwrap();
    // large forward error
    assert!((result - 0.682328).abs() < 1e-6, "root = {}", result);
}

#[test]
fn test_brent_1() {
    let _ = env_logger::try_init();
    let f = Polynomial::with_coefficients(vec![-4., 16., 35., -69., -102., 45., 54.].into());
    let ranges = [[-1.5, -1.], [0., 0.3], [0.3, 0.8], [0.8, 1.5]];
    let solvers = ranges.map(|r| BrentSolver::with_search_range(r).with_error_tolerance(1e-14));
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
    let f = Polynomial::<f32>::with_coefficients(vec![-4., 16., 35., -69., -102., 45., 54.].into());
    let ranges = [[-1.5, -1.], [0., 0.3], [0.3, 0.8], [0.8, 1.5]];
    let solvers = ranges.map(|r| BrentSolver::with_search_range(r).with_error_tolerance(1e-6));
    let roots = solvers.map(|s| s.solve(&f).unwrap());
    let backward_errors = roots.map(|x| f.eval(x));
    println!("roots = {:?}", roots);
    for i in 0..roots.len() {
        assert!(backward_errors[i].abs() < 1e-3,);
    }
}

#[test]
#[should_panic(expected = "Illegal initial guess")]
fn test_brent_illegal_range() {
    let f = Polynomial::<f32>::with_coefficients(vec![1., -2., 1.].into());
    let solver = BrentSolver::with_search_range([0., 2.]).with_error_tolerance(1e-6);
    solver.solve(&f);
}
