use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

type Inner = f64;

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct F64(Inner);

impl F64 {
    pub fn to_f64(&self) -> f64 {
        self.0
    }

    pub fn abs(&self) -> Self {
        f64::from_bits(self.0.to_bits() & (u64::MAX / 2)).into()
    }
}

impl From<Inner> for F64 {
    fn from(value: Inner) -> Self {
        F64(value)
    }
}

impl Add for F64 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.0.add(rhs.0).into()
    }
}

impl AddAssign for F64 {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0
    }
}

impl Sub for F64 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.0.sub(rhs.0).into()
    }
}

impl SubAssign for F64 {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0
    }
}

impl Mul for F64 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.0.mul(rhs.0).into()
    }
}

impl MulAssign for F64 {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0
    }
}

impl Div for F64 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self.0.div(rhs.0).into()
    }
}

impl DivAssign for F64 {
    fn div_assign(&mut self, rhs: Self) {
        self.0 /= rhs.0
    }
}

impl Neg for F64 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.0.neg().into()
    }
}
