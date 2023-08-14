use core_float::CoreFloat;

#[inline]
pub fn vec_zeros<T: CoreFloat>(len: usize) -> Vec<T> {
    vec![T::ZERO; len]
}
