#![no_std]
#![feature(asm_experimental_arch)]

mod patch;
mod shared_memory;

pub use patch::Patch;
pub use shared_memory::SharedMemory;

use core::{
    marker::PhantomData,
    ops::{Add, AddAssign, Index, IndexMut, Mul},
};

pub struct Image<P> {
    p: PhantomData<P>,
    ptr: *mut u8,
    width: usize,
    height: usize,
    pitch: usize,
}

impl<P> Image<P> {
    pub fn new(ptr: *mut u8, width: usize, height: usize, pitch: usize) -> Self {
        Self {
            p: PhantomData,
            ptr,
            height,
            width,
            pitch,
        }
    }

    pub fn get(&self, col: usize, row: usize) -> Option<P> {
        if col < self.width && row < self.height {
            unsafe {
                let i = row * self.pitch + col * core::mem::size_of::<P>();
                let ptr = self.ptr.add(i) as *mut P;
                Some(ptr.read())
            }
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, col: usize, row: usize) -> Option<&mut P> {
        if col < self.width && row < self.height {
            unsafe {
                let i = row * self.pitch + col * core::mem::size_of::<P>();
                let ptr = self.ptr.add(i) as *mut P;
                Some(&mut *ptr)
            }
        } else {
            None
        }
    }
}

impl<P> Index<(usize, usize)> for Image<P> {
    type Output = P;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (col, row) = index;
        assert!(col < self.width);
        assert!(row < self.height);
        unsafe {
            let i = row * self.pitch + col * core::mem::size_of::<P>();
            let ptr = self.ptr.add(i) as *mut P;
            &*ptr
        }
    }
}

impl<P> IndexMut<(usize, usize)> for Image<P> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (col, row) = index;
        assert!(col < self.width);
        assert!(row < self.height);
        unsafe {
            let i = col * core::mem::size_of::<P>() + row * self.pitch;
            let ptr = self.ptr.add(i) as *mut P;
            &mut *ptr
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Rgb<T> {
    pub r: T,
    pub g: T,
    pub b: T,
}

impl Mul<f32> for Rgb<f32> {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            r: self.r * rhs,
            g: self.g * rhs,
            b: self.b * rhs,
        }
    }
}

impl Add for Rgb<f32> {
    type Output = Self;

    fn add(self, Self { r, g, b }: Self) -> Self::Output {
        Self {
            r: self.r + r,
            g: self.g + g,
            b: self.b + b,
        }
    }
}

impl AddAssign for Rgb<f32> {
    fn add_assign(&mut self, Self { r, g, b }: Self) {
        self.r += r;
        self.g += g;
        self.b += b;
    }
}

impl From<Rgb<u8>> for Rgb<f32> {
    fn from(value: Rgb<u8>) -> Self {
        Self {
            r: value.r as f32 / u8::MAX as f32,
            g: value.g as f32 / u8::MAX as f32,
            b: value.b as f32 / u8::MAX as f32,
        }
    }
}

impl From<Rgb<f32>> for Rgb<u8> {
    fn from(value: Rgb<f32>) -> Self {
        Self {
            r: (value.r * u8::MAX as f32) as u8,
            g: (value.g * u8::MAX as f32) as u8,
            b: (value.b * u8::MAX as f32) as u8,
        }
    }
}
