use crate::Rgb;

pub trait SharedMemory
where
    Self: Sized,
{
    unsafe fn load(ptr: *const Self) -> Self;
    unsafe fn store(&self, ptr: *mut Self);
}

#[cfg(target_arch = "nvptx64")]
impl SharedMemory for u8 {
    unsafe fn load(ptr: *const Self) -> Self {
        let ptr = ptr as u16;
        let val;
        core::arch::asm!("ld.shared.u8 {}, [{}];", out(reg16) val, in(reg16) ptr);
        val
    }

    unsafe fn store(&self, ptr: *mut Self) {
        let ptr = ptr as u16;
        core::arch::asm!("st.shared.u8 [{}], {};", in(reg16) ptr, in(reg16) *self);
    }
}

#[cfg(not(target_arch = "nvptx64"))]
impl SharedMemory for u8 {
    unsafe fn load(_ptr: *const Self) -> Self {
        unimplemented!()
    }

    unsafe fn store(&self, _ptr: *mut Self) {
        unimplemented!()
    }
}

#[cfg(target_arch = "nvptx64")]
impl SharedMemory for f32 {
    unsafe fn load(ptr: *const Self) -> Self {
        let ptr = ptr as u16;
        let val;
        core::arch::asm!("ld.shared.u32 {}, [{}];", out(reg32) val, in(reg16) ptr);
        val
    }

    unsafe fn store(&self, ptr: *mut Self) {
        let ptr = ptr as u16;
        core::arch::asm!("st.shared.u32 [{}], {};", in(reg16) ptr, in(reg32) *self);
    }
}

#[cfg(not(target_arch = "nvptx64"))]
impl SharedMemory for f32 {
    unsafe fn load(_ptr: *const Self) -> Self {
        unimplemented!()
    }

    unsafe fn store(&self, _ptr: *mut Self) {
        unimplemented!()
    }
}

impl<T: SharedMemory> SharedMemory for Rgb<T> {
    unsafe fn load(ptr: *const Self) -> Self {
        let ptr = ptr as *const T;
        Self {
            r: T::load(ptr),
            g: T::load(ptr.add(1)),
            b: T::load(ptr.add(2)),
        }
    }

    unsafe fn store(&self, ptr: *mut Self) {
        let ptr = ptr as *mut T;
        self.r.store(ptr);
        self.g.store(ptr.add(1));
        self.b.store(ptr.add(2));
    }
}
