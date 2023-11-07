use numerical::{
    base_float::BaseFloat,
    dim1_func::SimpleDim1ContinuousFunc,
    dim1_solver::{BisectionSolver, Dim1Solver},
    polynomial::Polynomial,
};

#[test]
fn test_bisection_0() {
    let f = SimpleDim1ContinuousFunc::new(&|x: f64| 2. * x + 1.);
    let solver = BisectionSolver::with_search_range([-10., 10.]).with_error_tolerance(1e-5);
    let result = solver.solve(&f).unwrap();
    assert!((result + 0.5).abs() < 1e-3, "root = {}", result);
}

#[test]
fn test_bisection_1() {
    let f = Polynomial::with_coefficients(vec![2., -3., 1.].into());
    let solver_a = BisectionSolver::with_search_range([-10., 1.5]).with_error_tolerance(1e-5);
    let solver_b = BisectionSolver::with_search_range([1.5, 10.]).with_error_tolerance(1e-5);
    let result_a = solver_a.solve(&f).unwrap();
    let result_b = solver_b.solve(&f).unwrap();
    assert!((result_a - 1.).abs() < 1e-3, "root a = {}", result_a);
    assert!((result_b - 2.).abs() < 1e-3, "root b = {}", result_b);
}

#[test]
fn test_bisection_2() {
    let f = Polynomial::with_coefficients(vec![-8. / 27., 4. / 3., -2., 1.].into());
    let solver = BisectionSolver::with_search_range([0., 2.]).with_error_tolerance(1e-14);
    let result = solver.solve(&f).unwrap();
    // large forward error
    assert!((result - 2. / 3.).abs() > 1e-6, "root = {}", result);
}
