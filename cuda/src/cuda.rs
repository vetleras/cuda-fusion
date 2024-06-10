use std::{
    ptr,
    sync::{Arc, Mutex, Weak},
};

use crate::driver::{self, Result};
// When a context is created it is made current to the current thread (by the CUDA driver).
// This makes RAII difficult as creating one context, allocating some memory, creating a new context and use memory allocated in the first context would not work.
// We therefore must ensure that only one context lives in our program at a time. To ensure this we need some kind of static context,
// but we also need to run its constructor and destructor. We could use lazy static to run the constructor, but this would not ensure that the destructor is run.
// This is solved by the following implementation.
static CONTEXT: Mutex<Weak<Context>> = Mutex::new(Weak::new());
// The mutex is present to avoid a TOCTOU between failing to upgrade weak and creating a new context.

pub struct Cuda {
    context: Arc<Context>,
}

impl !Sync for Cuda {}
impl !Send for Cuda {}

impl Cuda {
    pub fn new() -> Result<Self> {
        let mut weak = CONTEXT.lock().unwrap();
        if let Some(context) = weak.upgrade() {
            context.make_current()?;
            Ok(Self { context })
        } else {
            let context = Arc::new(Context::new()?);
            *weak = Arc::downgrade(&context);
            Ok(Self { context })
        }
    }

    pub(crate) fn context(&self) -> driver::CUcontext {
        self.context.inner
    }

    pub fn get_alignment(&self) -> Result<usize> {
        let mut device_ptr = 0;
        let mut pitch = 0;
        unsafe {
            driver::cuMemAllocPitch_v2(&mut device_ptr, &mut pitch, 1, 2, 16).to_result()?;
            driver::cuMemFree_v2(device_ptr).to_result()?;
        }
        Ok(pitch)
    }
}

struct Context {
    inner: driver::CUcontext,
}

unsafe impl Sync for Context {}
unsafe impl Send for Context {}

impl Context {
    fn new() -> Result<Self> {
        unsafe {
            driver::cuInit(0).to_result()?;

            let mut device = 0;
            driver::cuDeviceGet(&mut device, 0).to_result()?;

            let mut inner = ptr::null_mut();
            driver::cuCtxCreate_v2(&mut inner, 0, device).to_result()?;

            Ok(Self { inner })
        }
    }

    fn make_current(&self) -> Result<()> {
        unsafe { driver::cuCtxSetCurrent(self.inner).to_result() }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            driver::cuCtxDestroy_v2(self.inner)
                .to_result()
                .expect("cuCtxDestroy failed while dropping Context");
        }
    }
}
