use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

type Inner = f64;

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct F64(Inner);

macro_rules! impl_ops_0 {
    ($trait: ty, $method: tt) => {
        impl $trait for F64 {
            type Output = Self;

            fn $method(self, rhs: Self) -> Self::Output {
                self.0.$method(rhs.0).into()
            }
        }

        impl $trait for &F64 {
            type Output = F64;

            fn $method(self, rhs: Self) -> Self::Output {
                self.0.$method(rhs.0).into()
            }
        }
    };
}

macro_rules! impl_ops_1 {
    ($trait: ty, $method: tt) => {
        impl $trait for F64 {
            fn $method(&mut self, rhs: Self) {
                self.0.$method(rhs.0)
            }
        }
    };
}

impl F64 {
    pub fn to_f64(&self) -> Inner {
        self.0
    }

    pub fn abs(&self) -> Self {
        Inner::from_bits(self.0.to_bits() & (u64::MAX / 2)).into()
    }

    pub fn map_vec(f64_vec: Vec<f64>) -> Vec<Self> {
        f64_vec.into_iter().map(Self::from).collect()
    }

    pub fn max(&self, rhs: Self) -> Self {
        self.0.max(rhs.0).into()
    }

    pub fn min(&self, rhs: Self) -> Self {
        self.0.min(rhs.0).into()
    }
}

impl From<Inner> for F64 {
    fn from(value: Inner) -> Self {
        F64(value)
    }
}

impl_ops_0!(Add, add);
impl_ops_0!(Sub, sub);
impl_ops_0!(Mul, mul);
impl_ops_0!(Div, div);

impl_ops_1!(AddAssign, add_assign);
impl_ops_1!(SubAssign, sub_assign);
impl_ops_1!(MulAssign, mul_assign);
impl_ops_1!(DivAssign, div_assign);

impl Neg for F64 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.0.neg().into()
    }
}

impl std::fmt::Display for F64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
