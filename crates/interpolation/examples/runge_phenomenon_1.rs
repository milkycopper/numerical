use floating_point::F64;
use interpolation::polynomial::{interpolated_polynomial, polynomial_interpolation};
use plotly::{Plot, Scatter};

pub fn main() {
    let mut plot = Plot::new();

    let inter_n = 16;
    let x_array = (0..(inter_n + 1))
        .map(|i| F64::from(-1.0 + 2.0 / inter_n as f64 * i as f64))
        .collect::<Vec<_>>();
    let y_array = x_array
        .iter()
        .map(|x| F64::from(1.0) / (F64::from(1.0) + F64::from(12.0) * *x * *x))
        .collect::<Vec<_>>();

    let coes =
        polynomial_interpolation(x_array.clone().into_iter().zip(y_array.clone().into_iter()));
    let base_points = (0..inter_n)
        .map(|i| F64::from(-1.0 + 2.0 / inter_n as f64 * i as f64))
        .collect::<Vec<_>>();

    let n = 300;
    let plot_x_array = (0..n)
        .map(|i| F64::from(-1.01 + 2.02 / n as f64 * i as f64))
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
