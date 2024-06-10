use crate::SharedMemory;

pub struct Patch<const N: usize, T: SharedMemory> {
    shared: *const T,
    block_width: usize,
    base_col: usize,
    base_row: usize,
}

impl<const N: usize, T: SharedMemory> Patch<N, T> {
    #[cfg(target_arch = "nvptx64")]
    pub unsafe fn new(
        shared: *const T,
        block_width: usize,
        base_col: usize,
        base_row: usize,
    ) -> Self {
        Self {
            shared,
            block_width,
            base_col,
            base_row,
        }
    }

    pub fn get(&self, col: usize, row: usize) -> T {
        assert!(col < N);
        assert!(row < N);
        let offset = self.base_col + col - N / 2 + (self.base_row + row - N / 2) * self.block_width;
        unsafe { T::load(self.shared.add(offset)) }
    }
}
