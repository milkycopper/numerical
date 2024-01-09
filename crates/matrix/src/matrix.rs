use core::ops::Index;

pub trait Matrix<T>: Index<(u32, u32), Output = T> {
    fn shape(&self) -> (u32, u32);

    fn row_count(&self) -> u32 {
        self.shape().0
    }

    fn col_count(&self) -> u32 {
        self.shape().1
    }

    fn is_square(&self) -> bool {
        self.shape().0 == self.shape().1
    }
}
