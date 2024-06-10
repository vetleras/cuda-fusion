use std::{marker::PhantomData, rc::Rc};

use interface::{Image, Patch};
use kernel::{MapImageKernel, MapPatchKernel, MapPixelKernel};

mod codegen;
mod compiler;
mod computational_dependency_graph;
mod pixel;
mod transformation;

use cdg::Operation;
use computational_dependency_graph as cdg;
use pixel::Pixel;
use transformation::Output;
pub use transformation::Transformation;

pub struct Node<P> {
    p: PhantomData<P>,
    inner: Rc<cdg::Node>,
}

pub fn new_input<P: Pixel>(name: String, width: usize, height: usize) -> Node<P> {
    Node {
        p: PhantomData,
        inner: Rc::new(cdg::Node::Input {
            name,
            width,
            height,
            pixel_type: P::ty(),
        }),
    }
}

impl<P: Pixel> Node<P> {
    fn new<T: Pixel>(operation: Operation) -> Node<T> {
        Node {
            p: PhantomData,
            inner: Rc::new(cdg::Node::Operation(operation)),
        }
    }

    pub fn flip(&self) -> Self {
        Self::new(cdg::Operation::Flip {
            dependency: self.inner.clone(),
        })
    }

    pub fn map_pixel<T: Pixel>(&self, kernel: &MapPixelKernel<P, T>) -> Node<T> {
        let f =
            syn::parse_str(kernel.src()).expect("kernel.src should be parseable as syn::ItemFn");

        Self::new(Operation::MapPixel {
            dependency: self.inner.clone(),
            f,
            pixel_type: T::ty(),
        })
    }

    pub fn map_patch<const N: usize, T: Pixel>(
        &self,
        kernel: &MapPatchKernel<Patch<N, P>, T>,
    ) -> Node<T> {
        let f =
            syn::parse_str(kernel.src()).expect("kernel.src should be parseable as syn::ItemFn");

        Self::new(Operation::MapPatch {
            dependency: self.inner.clone(),
            f,
            dimension: N,
            pixel_type: T::ty(),
        })
    }

    pub fn map_image<T: Pixel>(
        &self,
        kernel: &MapImageKernel<Image<P>, T>,
        width: usize,
        height: usize,
    ) -> Node<T> {
        let f =
            syn::parse_str(kernel.src()).expect("kernel.src should be parseable as syn::ItemFn");

        Self::new(Operation::MapImage {
            dependency: self.inner.clone(),
            f,
            width,
            height,
            pixel_type: T::ty(),
        })
    }

    pub fn h_concat(&self, right: &Self) -> Self {
        Self::new(cdg::Operation::HConcat {
            dependency_left: self.inner.clone(),
            dependency_right: right.inner.clone(),
        })
    }

    pub fn v_concat(&self, bottom: &Self) -> Self {
        Self::new(cdg::Operation::VConcat {
            dependency_top: self.inner.clone(),
            dependency_bottom: bottom.inner.clone(),
        })
    }

    pub fn into_output(self) -> Output {
        Output::new(self.inner)
    }
}
