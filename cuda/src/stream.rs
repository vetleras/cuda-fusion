use std::{marker::PhantomData, ptr};

use crate::{driver, Cuda, Result};

pub struct Stream<'a> {
    p: PhantomData<&'a Cuda>,
    inner: driver::CUstream,
}

impl<'a> Stream<'a> {
    pub fn new(_cuda: &'a Cuda) -> Result<Self> {
        let mut inner = ptr::null_mut();
        unsafe {
            driver::cuStreamCreate(&mut inner, 0).to_result()?;
        }
        
        Ok(Self {
            p: PhantomData,
            inner,
        })
    }

    pub(crate) fn inner(&self) -> driver::CUstream {
        self.inner
    }

    pub fn synchronize(&mut self) -> Result<()> {
        unsafe { driver::cuStreamSynchronize(self.inner).to_result() }
    }
}

impl<'a> Drop for Stream<'a> {
    fn drop(&mut self) {
        unsafe {
            driver::cuStreamDestroy_v2(self.inner)
                .to_result()
                .expect("cuStreamDestroy failed while dropping Stream")
        }
    }
}
