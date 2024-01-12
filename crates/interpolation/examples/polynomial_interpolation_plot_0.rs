use floating_point::F64;
use interpolation::polynomial::{interpolated_polynomial, polynomial_interpolation};
use plotly::{Plot, Scatter};

pub fn main() {
    let mut plot = Plot::new();

    let x_array = F64::map_vec(vec![-1., 0., 2., 3.]);
    let y_array = F64::map_vec(vec![5., -2., 3., -4.]);

    let coes =
        polynomial_interpolation(x_array.clone().into_iter().zip(y_array.clone().into_iter()));
    let base_points = F64::map_vec(vec![-1., 0., 2.]);

    let plot_x_array = (0..100)
        .map(|i| F64::from(-1.3 + 4.5 / 100.0 * i as f64))
        .collect::<Vec<_>>();
    let plot_y_array = plot_x_array
        .iter()
        .map(|x| interpolated_polynomial(*x, &coes, &base_points))
        .collect::<Vec<_>>();

    let trace = Scatter::new(
        x_array.iter().map(F64::to_f64).collect(),
        y_array.iter().map(F64::to_f64).collect(),
    );
    plot.add_trace(trace);
    let trace = Scatter::new(
        plot_x_array.iter().map(F64::to_f64).collect(),
        plot_y_array.iter().map(F64::to_f64).collect(),
    );
    plot.add_trace(trace);

    plot.write_html("out.html");
}
