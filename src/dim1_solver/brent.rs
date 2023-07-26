use super::Dim1Solver;
use crate::{base_float::BaseFloat, dim1_func::Dim1Func};
use core_float::core_float_traits::CoreFloat;

/// A [`BrentSolver`] which implemented the [`Dim1Solver`] using a hybrid
/// method to solve the root of a one-dimensional function.
///
/// `BrentSolver` is most desirable to combine the property of guaranteed convergence,
/// from the Bisection Method, with the property of quadratic convergence. Roughly speaking,
/// the Inverse Quadratic Interpolation method is attempted, if (1) the backward error improves
/// and (2) the bracketing interval is cut at least in half. If not, the Method of False Position
/// is attempted with the same goal. If it fails as well, a Bisection Method
/// step is taken, guaranteeing that the uncertainty is cut at least in half.
pub struct BrentSolver<T: PartialEq + PartialOrd> {
    search_range: [T; 2],
    error_tolerance: T,
}

impl<T: CoreFloat> BrentSolver<T> {
    /// BrentSolver constructor
    ///
    /// # Arguments
    ///
    /// * `search_range` - The root is expected be within this range `[a, b]`
    /// * `error_tolerance` - The error tolerance determines the accuracy of the root
    ///
    /// # Panic
    ///
    /// - Will panic if `search_range[0] >= search_range[1]`
    /// - Will panic if `error_tolerance <= T::EPSILON`, which means
    ///   the accuracy of the root will be limited by the machine epsilon
    pub fn new(search_range: [T; 2], error_tolerance: T) -> Self {
        assert!(search_range[0] < search_range[1]);
        assert!(error_tolerance > T::EPSILON);
        Self {
            search_range,
            error_tolerance,
        }
    }
}

impl<T: BaseFloat> Dim1Solver<T> for BrentSolver<T> {
    fn solve(&self, func: &impl Dim1Func<T>) -> Option<T> {
        let (a, b) = (self.search_range[0], self.search_range[1]);
        let mut xy_pairs = [a, (a + b).half(), b].map(|a| XYPair::new(a, func.eval(a)));

        assert!(
            xy_pairs[0].y.sig_ne(xy_pairs[2].y),
            "Illegal initial guess: {xy_pairs:?}"
        );

        let mut iter_num: usize = 0;

        while (xy_pairs[2].x - xy_pairs[0].x) > self.error_tolerance.double() {
            iter_num += 1;

            // Inverse Quadratic Interpolation
            let (mut max_error_index, mut max_error) = (0_usize, xy_pairs[0].y.abs());

            // decide max error point
            for (i, pair) in xy_pairs.iter().enumerate().skip(1) {
                let error = pair.y.abs();
                if error > max_error {
                    max_error_index = i;
                    max_error = error;
                }
            }

            if max_error_index != 1 {
                let next_x = inverse_quadratic_iterpolate(&xy_pairs);
                let next_y = func.eval(next_x);

                if next_y == T::ZERO {
                    log::info!("Brent solve, iteration number = {iter_num}");
                    return Some(next_x);
                } else if next_y.abs() < max_error {
                    let axis_crossed: bool;
                    let new_interval: T;
                    let replace_and_swap: bool;

                    if max_error_index == 0 && next_x > xy_pairs[1].x {
                        new_interval = xy_pairs[2].x - xy_pairs[1].x;
                        axis_crossed = xy_pairs[1].y.sig_ne(xy_pairs[2].y);
                        replace_and_swap = true;
                    } else if max_error_index == 2 && next_x < xy_pairs[1].x {
                        new_interval = xy_pairs[1].x - xy_pairs[0].x;
                        axis_crossed = xy_pairs[0].y.sig_ne(xy_pairs[1].y);
                        replace_and_swap = true;
                    } else if max_error_index == 0 {
                        new_interval = xy_pairs[2].x - next_x;
                        axis_crossed = next_y.sig_ne(xy_pairs[2].y);
                        replace_and_swap = false;
                    } else {
                        new_interval = next_x - xy_pairs[0].x;
                        axis_crossed = next_y.sig_ne(xy_pairs[0].y);
                        replace_and_swap = false;
                    }

                    let old_interval = xy_pairs[2].x - xy_pairs[0].x;

                    if axis_crossed && old_interval >= new_interval.double() {
                        let new_pair = XYPair::new(next_x, next_y);

                        // reorder
                        if replace_and_swap {
                            xy_pairs[max_error_index] = xy_pairs[1];
                            xy_pairs[1] = new_pair;
                        } else {
                            xy_pairs[max_error_index] = new_pair;
                        }

                        log::info!("Brent Solver, IQI, iteration index = {iter_num}");

                        continue;
                    }
                }
            }

            // Method of False Position
            let replaced_index: usize;
            let (next_x, old_interval) = if xy_pairs[0].y.sig_ne(xy_pairs[1].y) {
                replaced_index = 2;
                (secant_iter(&xy_pairs[..2]), xy_pairs[1].x - xy_pairs[0].x)
            } else {
                debug_assert!(xy_pairs[1].y.sig_ne(xy_pairs[2].y));
                replaced_index = 0;
                (secant_iter(&xy_pairs[1..]), xy_pairs[2].x - xy_pairs[1].x)
            };

            let next_y = func.eval(next_x);

            if next_y == T::ZERO {
                log::info!("Brent solve, iteration number = {iter_num}");
                return Some(next_x);
            }

            let new_interval = if replaced_index == 0 {
                if next_y.sig_eq(xy_pairs[1].y) {
                    xy_pairs[2].x - next_x
                } else {
                    next_x - xy_pairs[1].x
                }
            } else if next_y.sig_eq(xy_pairs[0].y) {
                xy_pairs[1].x - next_x
            } else {
                next_x - xy_pairs[0].x
            };

            if next_y.abs() < xy_pairs[replaced_index].y.abs()
                && old_interval > new_interval.double()
            {
                xy_pairs[replaced_index] = xy_pairs[1];
                xy_pairs[1] = XYPair::new(next_x, next_y);

                log::info!("Brent Solver, Method of False Position, iteration index = {iter_num}");
            } else {
                // Bisection
                let other_two_x = [1, 2].map(|i| xy_pairs[(replaced_index + i) % 3].x);
                let next_x = (other_two_x[0] + other_two_x[1]).half();
                let next_y = func.eval(next_x);
                if next_y == T::ZERO {
                    log::info!("Brent solve, iteration number = {}", iter_num);
                    return Some(next_x);
                } else {
                    xy_pairs[replaced_index] = xy_pairs[1];
                    xy_pairs[1] = XYPair::new(next_x, next_y);
                }

                log::info!("Brent Solver, Bisection, iteration index = {iter_num}");
            }
        }

        log::info!("Brent solve, iteration number = {}", iter_num);

        Some(xy_pairs[1].x)
    }
}

fn inverse_quadratic_iterpolate<T: CoreFloat>(xy_pairs: &[XYPair<T>; 3]) -> T {
    let one = T::ONE;
    let (a, fa) = (xy_pairs[0].x, xy_pairs[0].y);
    let (b, fb) = (xy_pairs[1].x, xy_pairs[1].y);
    let (c, fc) = (xy_pairs[2].x, xy_pairs[2].y);

    let q = fa / fb;
    let r = fc / fb;
    let s = fc / fa;
    c - (r * (r - q) * (c - b) + (one - r) * s * (c - a)) / ((q - one) * (r - one) * (s - one))
}

// Method of False Position
fn secant_iter<T: CoreFloat>(xy_pairs: &[XYPair<T>]) -> T {
    debug_assert!(xy_pairs.len() == 2);
    let (a, fa) = (xy_pairs[0].x, xy_pairs[0].y);
    let (b, fb) = (xy_pairs[1].x, xy_pairs[1].y);
    // assume b > a
    (b * fa - a * fb) / (fa - fb)
}

#[derive(Clone, Copy, Debug)]
struct XYPair<T> {
    pub x: T,
    pub y: T,
}

impl<T> XYPair<T> {
    pub fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
}
