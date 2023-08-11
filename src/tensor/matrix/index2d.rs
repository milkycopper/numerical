/// Index for 2D matrix
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Index2D {
    pub row: usize,
    pub col: usize,
}

impl Index2D {
    /// flatten 2D index of full `M * N` matrix to 1D in row by row sequence
    #[inline]
    pub const fn full_to_1d(&self, col_size: usize) -> usize {
        debug_assert!(self.col < col_size);
        self.row * col_size + self.col
    }

    /// flatten 2D index of size `N` lower triangular matrix to 1D in row by row sequence
    ///
    /// # Notice
    ///
    /// To get meaningful results, ensure that `self.row >= self.col`
    #[inline]
    pub fn lt_to_1d(&self) -> usize {
        debug_assert!(
            self.row >= self.col,
            "row = {}, col = {}",
            self.row,
            self.col
        );
        self.row * (self.row + 1) / 2 + self.col
    }

    /// flatten 2D index of size `N` upper triangular matrix to 1D in row by row sequence
    ///
    /// # Notice
    ///
    /// To get meaningful results, ensure that `self.row <= self.col`
    #[inline]
    pub const fn ut_to_1d(&self, matrix_size: usize) -> usize {
        debug_assert!(self.row <= self.col);
        debug_assert!(self.row < matrix_size);
        debug_assert!(self.col < matrix_size);
        (matrix_size * 2 - self.row - 1) * self.row / 2 + self.col
    }

    /// self.row * self.col
    #[inline]
    pub const fn area_size(&self) -> usize {
        self.row * self.col
    }

    /// exchange row and col
    #[inline]
    pub const fn transpose(&self) -> Self {
        Self {
            row: self.col,
            col: self.row,
        }
    }
}

impl From<(usize, usize)> for Index2D {
    fn from(value: (usize, usize)) -> Self {
        Index2D {
            row: value.0,
            col: value.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_to_1d() {
        assert!(Index2D::from((3, 4)).full_to_1d(5) == 19);
    }

    #[test]
    fn test_lt_to_1d() {
        assert!(Index2D::from((4, 3)).lt_to_1d() == 13);
        assert!(Index2D::from((5, 3)).lt_to_1d() == 18);
    }

    #[test]
    fn test_ut_to_1d() {
        assert!(Index2D::from((4, 5)).ut_to_1d(6) == 19);
    }
}
