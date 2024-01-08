use core::ops::{Add, Div, Mul, Neg, Sub};

type Inner = f64;

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct F64(Inner);

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

impl Sub for F64 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.0.sub(rhs.0).into()
    }
}

impl Mul for F64 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.0.mul(rhs.0).into()
    }
}

impl Div for F64 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self.0.div(rhs.0).into()
    }
}

impl Neg for F64 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.0.neg().into()
    }
}
