/// Base float functions including those not supported in core
pub trait BaseFloat: core_float::core_float_traits::CoreFloat {
    fn abs(self) -> Self;
    fn signum(self) -> Self;

    #[inline]
    fn sig_ne(self, other: Self) -> bool {
        self.signum() * other.signum() == -Self::ONE
    }
    #[inline]
    fn sig_eq(self, other: Self) -> bool {
        self.signum() * other.signum() == Self::ONE
    }
}

impl BaseFloat for f32 {
    #[inline]
    fn abs(self) -> Self {
        <f32>::abs(self)
    }
    #[inline]
    fn signum(self) -> Self {
        <f32>::signum(self)
    }
}

impl BaseFloat for f64 {
    #[inline]
    fn abs(self) -> Self {
        <f64>::abs(self)
    }
    #[inline]
    fn signum(self) -> Self {
        <f64>::signum(self)
    }
}
