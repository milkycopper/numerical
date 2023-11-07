use super::{Dim1Solver, SolveResult};
use crate::{base_float::BaseFloat, dim1_func::Dim1ContinuousFunc};
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
pub struct BrentSolver<T> {
    search_range: [T; 2],
    error_tolerance: T,
}

impl<T: CoreFloat> BrentSolver<T> {
    pub fn with_search_range(range: [T; 2]) -> Self {
        assert!(range[0] < range[1]);
        Self {
            search_range: range,
            error_tolerance: T::EPSILON,
        }
    }

    pub fn with_error_tolerance(mut self, tolerance: T) -> Self {
        self.error_tolerance = tolerance;
        self
    }
}

impl<T: BaseFloat> Dim1Solver<T> for BrentSolver<T> {
    fn solve<F>(&self, func: &F) -> SolveResult<T>
    where
        F: Dim1ContinuousFunc<T> + ?Sized,
    {
        const FIRST: usize = 0;
        const MIDDLE: usize = 1;
        const LAST: usize = 2;

        let (a, b) = (self.search_range[0], self.search_range[1]);
        let mut xy_pairs = [a, (a + b).half(), b].map(|a| XYPair::new(a, func.eval(a)));

        assert!(
            xy_pairs[FIRST].y.sig_ne(xy_pairs[LAST].y),
            "Illegal initial guess: {xy_pairs:?}"
        );

        let mut iter_count: usize = 0;

        while (xy_pairs[LAST].x - xy_pairs[FIRST].x) > self.error_tolerance.double() {
            iter_count += 1;

            // Inverse Quadratic Interpolation
            let (mut max_error_index, mut max_error) = (FIRST, xy_pairs[FIRST].y.abs());

            // decide max error point
            for (i, pair) in xy_pairs.iter().enumerate().skip(1) {
                let error = pair.y.abs();
                if error > max_error {
                    max_error_index = i;
                    max_error = error;
                }
            }

            if max_error_index != MIDDLE {
                let next_x = inverse_quadratic_iterpolate(&xy_pairs);
                let next_y = func.eval(next_x);

                if next_y == T::ZERO {
                    return SolveResult {
                        solution: Some(next_x),
                        iter_count,
                    };
                } else if next_y.abs() < max_error {
                    let (new_interval, axis_crossed, replace_and_swap) =
                        if max_error_index == FIRST && next_x > xy_pairs[MIDDLE].x {
                            (
                                xy_pairs[LAST].x - xy_pairs[MIDDLE].x,
                                xy_pairs[LAST].y.sig_ne(xy_pairs[MIDDLE].y),
                                true,
                            )
                        } else if max_error_index == LAST && next_x < xy_pairs[MIDDLE].x {
                            (
                                xy_pairs[MIDDLE].x - xy_pairs[FIRST].x,
                                xy_pairs[MIDDLE].y.sig_ne(xy_pairs[FIRST].y),
                                true,
                            )
                        } else if max_error_index == FIRST {
                            (
                                xy_pairs[LAST].x - next_x,
                                next_y.sig_ne(xy_pairs[LAST].y),
                                false,
                            )
                        } else {
                            (
                                next_x - xy_pairs[FIRST].x,
                                next_y.sig_ne(xy_pairs[FIRST].y),
                                false,
                            )
                        };

                    let old_interval = xy_pairs[LAST].x - xy_pairs[FIRST].x;

                    if axis_crossed && old_interval >= new_interval.double() {
                        let new_pair = XYPair::new(next_x, next_y);

                        // reorder
                        if replace_and_swap {
                            xy_pairs[max_error_index] = xy_pairs[MIDDLE];
                            xy_pairs[MIDDLE] = new_pair;
                        } else {
                            xy_pairs[max_error_index] = new_pair;
                        }

                        log::info!("Brent Solver, IQI, iteration index = {iter_count}");

                        continue;
                    }
                }
            }

            // Method of False Position
            let (replaced_index, next_x, old_interval) =
                if xy_pairs[FIRST].y.sig_ne(xy_pairs[MIDDLE].y) {
                    (
                        LAST,
                        secant_iter(&xy_pairs[FIRST..=MIDDLE]),
                        xy_pairs[MIDDLE].x - xy_pairs[FIRST].x,
                    )
                } else {
                    debug_assert!(xy_pairs[MIDDLE].y.sig_ne(xy_pairs[LAST].y));
                    (
                        FIRST,
                        secant_iter(&xy_pairs[MIDDLE..=LAST]),
                        xy_pairs[LAST].x - xy_pairs[MIDDLE].x,
                    )
                };

            let next_y = func.eval(next_x);

            if next_y == T::ZERO {
                return SolveResult {
                    solution: Some(next_x),
                    iter_count,
                };
            }

            let new_interval = if replaced_index == FIRST {
                if next_y.sig_eq(xy_pairs[MIDDLE].y) {
                    xy_pairs[LAST].x - next_x
                } else {
                    next_x - xy_pairs[MIDDLE].x
                }
            } else if next_y.sig_eq(xy_pairs[FIRST].y) {
                xy_pairs[MIDDLE].x - next_x
            } else {
                next_x - xy_pairs[FIRST].x
            };

            if next_y.abs() < xy_pairs[replaced_index].y.abs()
                && old_interval >= new_interval.double()
            {
                xy_pairs[replaced_index] = xy_pairs[MIDDLE];
                xy_pairs[MIDDLE] = XYPair::new(next_x, next_y);

                log::info!(
                    "Brent Solver, Method of False Position, iteration index = {iter_count}"
                );
            } else {
                // Bisection
                let other_two_x = [1, 2].map(|i| xy_pairs[(replaced_index + i) % 3].x);
                let next_x = (other_two_x[0] + other_two_x[1]).half();
                let next_y = func.eval(next_x);
                if next_y == T::ZERO {
                    return SolveResult {
                        solution: Some(next_x),
                        iter_count,
                    };
                } else {
                    xy_pairs[replaced_index] = xy_pairs[MIDDLE];
                    xy_pairs[MIDDLE] = XYPair::new(next_x, next_y);
                }

                log::info!("Brent Solver, Bisection, iteration index = {iter_count}");
            }
        }

        SolveResult {
            solution: Some(xy_pairs[1].x),
            iter_count,
        }
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
