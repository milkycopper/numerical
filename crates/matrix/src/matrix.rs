use core::ops::Index;
use std::fmt::Display;

pub trait Matrix<T: Display>: Index<(u32, u32), Output = T> {
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

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "\n[")?;
        for i in 0..self.row_count() {
            write!(f, "[")?;
            for j in 0..self.col_count() {
                write!(f, "{}", self[(i, j)])?;
                if j != self.col_count() - 1 {
                    write!(f, ",")?;
                }
            }
            write!(f, "]")?;
            if i != self.row_count() - 1 {
                writeln!(f, ",")?;
            }
        }
        write!(f, "]")?;

        Ok(())
    }
}
