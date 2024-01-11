use core::ops::Index;

pub trait Matrix<T: std::fmt::Display>: Index<(usize, usize), Output = T> {
    fn shape(&self) -> (usize, usize);

    fn row_count(&self) -> usize {
        self.shape().0
    }

    fn col_count(&self) -> usize {
        self.shape().1
    }

    fn is_square(&self) -> bool {
        self.shape().0 == self.shape().1
    }

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
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
