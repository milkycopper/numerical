use floating_point::F64;

fn add_point(new_point: (F64, F64), points: &mut Vec<(F64, F64)>, states: &mut Vec<F64>) -> F64 {
    let (x, y) = new_point;
    let n = points.len();

    let mut new = y;
    for i in 0..n {
        let old = states[i];
        states[i] = new;
        new = (new - old) / (x - points[n - i - 1].0);
    }

    states.push(new);
    points.push(new_point);

    new
}

pub fn polynomial_interpolation<T: Iterator<Item = (F64, F64)>>(points: T) -> Vec<F64> {
    let mut coes = vec![];
    let mut temp_states = vec![];
    let mut temp_points = vec![];
    for p in points {
        coes.push(add_point(p, &mut temp_points, &mut temp_states));
    }
    coes
}

pub fn interpolated_polynomial(x: F64, coes: &Vec<F64>, base_points: &Vec<F64>) -> F64 {
    assert!(coes.len() == base_points.len() + 1);
    let n = coes.len();
    let mut y = coes[n - 1];
    for i in (0..(n - 1)).rev() {
        y = y * (x - base_points[i]) + coes[i];
    }

    y
}

#[cfg(test)]
mod tests {
    use floating_point::F64;

    use super::{add_point, polynomial_interpolation};

    #[test]
    fn test_add_point_0() {
        let mut points = vec![];
        let mut states = vec![];

        let coe = add_point((0.0.into(), 1.0.into()), &mut points, &mut states);
        assert!(coe == 1.0.into());

        let coe = add_point((2.0.into(), 2.0.into()), &mut points, &mut states);
        assert!(coe == 0.5.into());

        let coe = add_point((3.0.into(), 4.0.into()), &mut points, &mut states);
        assert!(coe == 0.5.into());

        let coe = add_point((1.0.into(), 0.0.into()), &mut points, &mut states);
        assert!(coe == (-0.5).into());
    }

    #[test]
    fn test_add_point_1() {
        let mut points = vec![];
        let mut states = vec![];

        let coe = add_point(((-1.0).into(), (-5.0).into()), &mut points, &mut states);
        assert!(coe == (-5.0).into());

        let coe = add_point((0.0.into(), (-1.0).into()), &mut points, &mut states);
        assert!(coe == 4.0.into());

        let coe = add_point((2.0.into(), 1.0.into()), &mut points, &mut states);
        assert!(coe == (-1.0).into());

        let coe = add_point((3.0.into(), 11.0.into()), &mut points, &mut states);
        assert!(coe == 1.0.into());
    }

    #[test]
    fn test_polynomial_interpolation() {
        let points = F64::map_vec(vec![-1., 0., 2., 3.])
            .into_iter()
            .zip(F64::map_vec(vec![-5., -1., 1., 11.]).into_iter());
        let coes = polynomial_interpolation(points);

        assert!(coes[0] == (-5.0).into());
        assert!(coes[1] == 4.0.into());
        assert!(coes[2] == (-1.0).into());
        assert!(coes[3] == 1.0.into());
    }
}
