use std::marker::PhantomData;

pub struct MapPixelKernel<A, B> {
    a: PhantomData<A>,
    b: PhantomData<B>,
    src: &'static str,
}

impl<A, B> MapPixelKernel<A, B> {
    #[doc(hidden)]
    pub const fn new(src: &'static str) -> Self {
        Self {
            a: PhantomData,
            b: PhantomData,
            src,
        }
    }

    pub fn src(&self) -> &'static str {
        self.src
    }
}

pub struct MapPatchKernel<A, B> {
    a: PhantomData<A>,
    b: PhantomData<B>,
    src: &'static str,
}

impl<A, B> MapPatchKernel<A, B> {
    #[doc(hidden)]
    pub const fn new(src: &'static str) -> Self {
        Self {
            a: PhantomData,
            b: PhantomData,
            src,
        }
    }

    pub fn src(&self) -> &'static str {
        self.src
    }
}

pub struct MapImageKernel<A, B> {
    a: PhantomData<A>,
    b: PhantomData<B>,
    src: &'static str,
}

impl<A, B> MapImageKernel<A, B> {
    #[doc(hidden)]
    pub const fn new(src: &'static str) -> Self {
        Self {
            a: PhantomData,
            b: PhantomData,
            src,
        }
    }

    pub fn src(&self) -> &'static str {
        self.src
    }
}
