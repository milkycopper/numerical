pub trait Dim1Func<T> {
    /// Evaluate `f(x)` at `x`
    fn eval(&self, x: T) -> T;
}

impl<T, F> Dim1Func<T> for F
where
    F: Fn(T) -> T,
{
    fn eval(&self, x: T) -> T {
        self(x)
    }
}
