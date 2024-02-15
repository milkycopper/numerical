use crate::F64;

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
type Inner = F64;

#[derive(Copy, Clone, PartialEq)]
pub struct Complex64 {
    pub real: Inner,
    pub imag: Inner,
}

impl Complex64 {
    pub const ZERO: Self = Self {
        real: F64::ZERO,
        imag: F64::ZERO,
    };

    pub const ONE: Self = Self {
        real: F64::ONE,
        imag: F64::ZERO,
    };

    pub const I: Self = Self {
        real: F64::ZERO,
        imag: F64::ONE,
    };

    pub fn new(real: f64, imag: f64) -> Self {
        Self {
            real: real.into(),
            imag: imag.into(),
        }
    }

    pub fn conjugate(&self) -> Self {
        Self {
            real: self.real,
            imag: self.imag.neg(),
        }
    }

    pub fn omega_n(n: usize) -> Self {
        let angle = core::f64::consts::TAU / n as f64;
        Self::new(angle.cos(), -angle.sin())
    }

    pub fn omega_n_power(n: usize, pow: usize) -> Self {
        let angle = ((pow % n) as f64 / n as f64) * core::f64::consts::TAU;
        Self::new(angle.cos(), -angle.sin())
    }
}

impl From<(Inner, Inner)> for Complex64 {
    fn from(value: (Inner, Inner)) -> Self {
        Self {
            real: value.0,
            imag: value.1,
        }
    }
}

impl From<(f64, f64)> for Complex64 {
    fn from(value: (f64, f64)) -> Self {
        Self {
            real: value.0.into(),
            imag: value.1.into(),
        }
    }
}

impl Add for Complex64 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            real: self.real + rhs.real,
            imag: self.imag + rhs.imag,
        }
    }
}

impl Sub for Complex64 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            real: self.real - rhs.real,
            imag: self.imag - rhs.imag,
        }
    }
}

impl Mul for Complex64 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            real: self.real * rhs.real - self.imag * rhs.imag,
            imag: self.real * rhs.imag + self.imag * rhs.real,
        }
    }
}

impl Div for Complex64 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let x = rhs.real * rhs.real + rhs.imag * rhs.imag;
        Self {
            real: (self.real * rhs.real + self.imag * rhs.imag) / x,
            imag: (self.imag * rhs.real - self.real * rhs.imag) / x,
        }
    }
}

impl AddAssign for Complex64 {
    fn add_assign(&mut self, rhs: Self) {
        self.real += rhs.real;
        self.imag += rhs.imag;
    }
}

impl SubAssign for Complex64 {
    fn sub_assign(&mut self, rhs: Self) {
        self.real -= rhs.real;
        self.imag -= rhs.imag;
    }
}

impl MulAssign for Complex64 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl DivAssign for Complex64 {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl Neg for Complex64 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            real: -self.real,
            imag: -self.imag,
        }
    }
}

impl core::iter::Sum for Complex64 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut sum = Complex64::ZERO;
        for item in iter {
            sum += item;
        }
        sum
    }
}

impl core::fmt::Debug for Complex64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:#?} {} {:#?} I",
            self.real,
            if self.imag > F64::ZERO { '+' } else { '-' },
            self.imag.abs()
        )
    }
}

impl core::fmt::Display for Complex64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} + {} I", self.real, self.imag)
    }
}

#[cfg(feature = "approx")]
impl approx::AbsDiffEq for Complex64 {
    type Epsilon = F64;

    fn default_epsilon() -> Self::Epsilon {
        F64::from(f64::EPSILON)
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.real
            .to_f64()
            .abs_diff_eq(&other.real.to_f64(), epsilon.to_f64())
            && self
                .imag
                .to_f64()
                .abs_diff_eq(&other.imag.to_f64(), epsilon.to_f64())
    }
}
