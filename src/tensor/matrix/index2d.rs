/// Index for 2D matrix
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Index2D {
    pub row: usize,
    pub col: usize,
}

impl Index2D {
    /// flatten 2D index to 1D in row by row sequence
    #[inline]
    pub const fn to_1d(&self, col_size: usize) -> usize {
        self.row * col_size + self.col
    }

    /// self.row * self.col
    #[inline]
    pub const fn area_size(&self) -> usize {
        self.row * self.col
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
