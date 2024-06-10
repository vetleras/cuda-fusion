use std::{
    ffi::{c_void, CString},
    ptr,
};

use crate::{driver, Result};

pub struct Function {
    pub inner: driver::CUfunction,
}

#[derive(Debug)]
pub struct Module {
    inner: driver::CUmodule,
}

impl Module {
    pub fn from_ptx(ptx: &str) -> driver::Result<Self> {
        let mut inner = ptr::null_mut();
        let ptx = CString::new(ptx).unwrap();
        unsafe {
            driver::cuModuleLoadData(&mut inner, ptx.as_ptr() as *const c_void)
                .to_result()
                .map(|_| Self { inner })
        }
    }

    pub fn get_function(&self, name: &str) -> driver::Result<Function> {
        let mut inner = ptr::null_mut();
        let name = CString::new(name).unwrap();
        unsafe {
            driver::cuModuleGetFunction(&mut inner, self.inner, name.as_ptr())
                .to_result()
                .map(|_| Function { inner })
        }
    }

    pub fn destroy(&self) -> Result<()> {
        unsafe { driver::cuModuleUnload(self.inner).to_result() }
    }
}
