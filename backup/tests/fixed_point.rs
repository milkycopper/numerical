use numerical::{
    dim1_func::SimpleDim1ContinuousFunc,
    dim1_solver::{Dim1Solver, FixedPointSolver},
};

#[test]
fn test_fixed_point_0() {
    let g = SimpleDim1ContinuousFunc::new(&|x: f64| 1. / x - x / 2.);
    let solver = FixedPointSolver::with_start_point(10.).with_error_tolerance(1e-14);
    let result = solver.solve(&g).unwrap();
    assert!((result - 2.0_f64.sqrt()).abs() < 1e-3, "root = {}", result);
}

#[test]
fn test_fixed_point_1() {
    let g = SimpleDim1ContinuousFunc::new(&|x: f64| x.cos() - x);
    let solver = FixedPointSolver::with_start_point(10.).with_error_tolerance(1e-14);
    let result = solver.solve(&g).unwrap();
    assert!(g(result).abs() < 1e-10, "root = {}", result);
}

#[test]
fn test_fixed_point_2() {
    let g = SimpleDim1ContinuousFunc::new(&|x: f64| 2.5 * x - 2.5);
    let solver = FixedPointSolver::with_start_point(10.3).with_error_tolerance(1e-14);
    let result = solver.solve(&g);
    assert!(result.is_none());
}
