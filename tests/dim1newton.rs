use numerical::{
    dim1_func::SimpleDim1ContinuousFunc, dim1_solver::Dim1NewtonSolver, polynomial::Polynomial,
};

#[test]
fn test_dim1_newton_0() {
    let g = SimpleDim1ContinuousFunc::new(&|x: f64| 1. / x - x / 2.);
    let dg = SimpleDim1ContinuousFunc::new(&|x: f64| -1. / (x * x) - 0.5);
    let solver = Dim1NewtonSolver::new(10., &g, &dg).with_error_tolerance(1e-14);
    let result = solver.solve().unwrap();
    assert!((result - 2.0_f64.sqrt()).abs() < 1e-8, "root = {}", result);
}

#[test]
fn test_dim1_newton_1() {
    let g = Polynomial::with_coefficients(vec![-1., 1., 0., 1.].into());
    let dg = SimpleDim1ContinuousFunc::new(&|x: f64| 3. * (x * x) + 1.);
    let solver = Dim1NewtonSolver::new(10., &g, &dg).with_error_tolerance(1e-14);
    let result = solver.solve().unwrap();
    assert!((result - 0.68232780).abs() < 1e-8, "root = {}", result);
}

#[test]
fn test_dim1_newton_2() {
    let g = Polynomial::with_coefficients(vec![-11. / 4., 0., -6., 0., 4.].into());
    let dg = SimpleDim1ContinuousFunc::new(&|x: f64| 16. * (x * x * x) - 12. * x);
    let solver = Dim1NewtonSolver::new(0.5, &g, &dg).with_error_tolerance(1e-14);
    assert!(solver.solve().is_none());
    let solver = Dim1NewtonSolver::new(0.65, &g, &dg).with_error_tolerance(1e-14);
    assert!(!solver.solve().is_none());
}
