pub trait BaseFloat: core_float::core_float_traits::CoreFloat {
    fn abs(self) -> Self;
    fn signum(self) -> Self;
}

impl BaseFloat for f32 {
    fn abs(self) -> Self {
        <f32>::abs(self)
    }

    fn signum(self) -> Self {
        <f32>::signum(self)
    }
}

impl BaseFloat for f64 {
    fn abs(self) -> Self {
        <f64>::abs(self)
    }

    fn signum(self) -> Self {
        <f64>::signum(self)
    }
}
