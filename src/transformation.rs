use std::{cell::UnsafeCell, collections::HashMap, rc::Rc};

use cuda::{
    graph::{DevicePtr, ExecutableGraph, Graph, MemCpyDirection},
    module::Module,
    stream::Stream,
    Cuda, Result,
};
use image::DynamicImage;
use interface::Rgb;
use itertools::Itertools;

use crate::{
    codegen,
    compiler::compile,
    computational_dependency_graph as cdg,
    pixel::{Pixel, PixelType},
};
use cdg::{toposort, Node, Operation};

struct Buffer {
    width: usize,
    height: usize,
    pixel_type: PixelType,
    inner: Rc<Box<UnsafeCell<[u8]>>>,
}

type ImageBuffer<P> = image::ImageBuffer<
    <P as Pixel>::ImageCratePixel,
    Vec<<<P as Pixel>::ImageCratePixel as image::Pixel>::Subpixel>,
>;

impl Buffer {
    unsafe fn copy_from_dynamic_image(&mut self, image: &DynamicImage) {
        assert_eq!(self.height, image.height() as usize);
        assert_eq!(self.width, image.width() as usize);

        fn copy<P: Pixel>(src: &ImageBuffer<P>, dst: &mut Buffer) {
            assert_eq!(dst.pixel_type, P::ty());
            let dst_slice = unsafe {
                std::slice::from_raw_parts_mut(dst.inner.get() as *mut P, dst.height * dst.width)
            };

            for (src_px, dst_px) in std::iter::zip(src.pixels(), dst_slice) {
                *dst_px = P::from_image_crate_pixel(*src_px);
            }
        }

        match image {
            DynamicImage::ImageRgb8(ib) => copy::<Rgb<u8>>(ib, self),
            DynamicImage::ImageRgb32F(ib) => copy::<Rgb<f32>>(ib, self),
            _ => unimplemented!(),
        };
    }

    unsafe fn to_dynamic_image(&self) -> DynamicImage {
        fn to_image_buffer<P: Pixel>(buffer: &Buffer) -> ImageBuffer<P> {
            assert_eq!(buffer.pixel_type, P::ty());
            let pixels = unsafe {
                std::slice::from_raw_parts(
                    buffer.inner.get() as *const P,
                    buffer.height * buffer.width,
                )
            };

            ImageBuffer::<P>::from_fn(buffer.width as u32, buffer.height as u32, |x, y| {
                pixels[x as usize + y as usize * buffer.width].into_image_crate_pixel()
            })
        }

        match self.pixel_type {
            PixelType::RgbU8 => DynamicImage::ImageRgb8(to_image_buffer::<Rgb<u8>>(self)),
            PixelType::RgbF32 => DynamicImage::ImageRgb32F(to_image_buffer::<Rgb<f32>>(self)),
        }
    }
}

pub struct Output(Rc<cdg::Node>);

impl Output {
    pub fn new(node: Rc<cdg::Node>) -> Self {
        Self(node)
    }
}

pub struct Transformation<'a> {
    input_buffers: HashMap<String, Buffer>,
    output_buffers: HashMap<String, Buffer>,
    executable_graph: ExecutableGraph<'a>,
    stream: Stream<'a>,
}

impl<'a> Transformation<'a> {
    pub fn new(cuda: &'a Cuda, outputs: HashMap<String, Output>) -> Result<Self> {
        let graph = Graph::new(&cuda).unwrap();

        let alignment = cuda.get_alignment()?;

        let mut input_buffers = HashMap::new();
        let mut graph_nodes: HashMap<*const Node, cuda::graph::Node> = HashMap::new();
        let mut device_ptrs: HashMap<*const Node, DevicePtr> = HashMap::new();

        let outputs: HashMap<String, &Node> = outputs
            .iter()
            .map(|(name, Output(node))| (name.clone(), &**node))
            .collect();

        for node in toposort(outputs.values().copied().collect_vec()) {
            let (alloc_node, device_ptr) = graph
                .add_mem_alloc_node(node.height(), node.pitch(alignment))
                .unwrap();

            let graph_node = match node {
                Node::Input {
                    name,
                    width,
                    height,
                    pixel_type,
                } => {
                    let (graph_node, buffer) = graph
                        .add_mem_cpy_node(
                            &alloc_node,
                            MemCpyDirection::HostToDevice,
                            *width,
                            *height,
                            node.pitch(alignment),
                            pixel_type.layout(),
                            device_ptr,
                        )
                        .unwrap();
                    let buffer = Buffer {
                        inner: buffer,
                        width: *width,
                        height: *height,
                        pixel_type: *pixel_type,
                    };
                    assert!(input_buffers.insert(name.clone(), buffer).is_none());
                    graph_node
                }

                Node::Operation(operation) => {
                    let block_width = 16;
                    let block_height = 16;

                    let f = match operation {
                        Operation::MapPixel {
                            dependency,
                            f,
                            pixel_type,
                        } => codegen::map_pixel(
                            device_ptrs[&Rc::as_ptr(&dependency)].inner(),
                            device_ptr.inner(),
                            dependency.width(),
                            dependency.height(),
                            dependency.pitch(alignment),
                            node.pitch(alignment),
                            dependency.pixel_type(),
                            *pixel_type,
                            f,
                            block_width,
                            block_height,
                        ),

                        Operation::MapPatch {
                            dependency,
                            f,
                            dimension,
                            pixel_type,
                        } => codegen::map_patch(
                            device_ptrs[&Rc::as_ptr(&dependency)].inner(),
                            device_ptr.inner(),
                            dependency.width(),
                            dependency.height(),
                            dependency.pitch(alignment),
                            node.pitch(alignment),
                            dependency.pixel_type(),
                            *pixel_type,
                            f,
                            *dimension,
                            block_width,
                            block_height,
                        ),

                        Operation::MapImage {
                            dependency,
                            f,
                            height,
                            width,
                            pixel_type,
                        } => codegen::map_image(
                            device_ptrs[&Rc::as_ptr(&dependency)].inner(),
                            device_ptr.inner(),
                            dependency.width(),
                            *width,
                            dependency.height(),
                            *height,
                            dependency.pitch(alignment),
                            node.pitch(alignment),
                            dependency.pixel_type(),
                            *pixel_type,
                            f,
                            block_width,
                            block_height,
                        ),

                        Operation::Flip { dependency } => codegen::flip(
                            device_ptrs[&Rc::as_ptr(&dependency)].inner(),
                            device_ptr.inner(),
                            dependency.width(),
                            dependency.height(),
                            dependency.pitch(alignment),
                            dependency.pixel_type(),
                            block_width,
                            block_height,
                        ),

                        Operation::HConcat {
                            dependency_left,
                            dependency_right,
                        } => codegen::h_concat(
                            device_ptrs[&Rc::as_ptr(&dependency_left)].inner(),
                            device_ptrs[&Rc::as_ptr(&dependency_right)].inner(),
                            device_ptr.inner(),
                            dependency_left.width(),
                            dependency_right.width(),
                            node.width(),
                            node.height(),
                            dependency_left.pitch(alignment),
                            dependency_right.pitch(alignment),
                            node.pitch(alignment),
                            node.pixel_type(),
                            block_width,
                            block_height,
                        ),

                        Operation::VConcat {
                            dependency_top,
                            dependency_bottom,
                        } => codegen::v_concat(
                            device_ptrs[&Rc::as_ptr(&dependency_top)].inner(),
                            device_ptrs[&Rc::as_ptr(&dependency_bottom)].inner(),
                            device_ptr.inner(),
                            node.width(),
                            dependency_top.height(),
                            dependency_bottom.height(),
                            node.height(),
                            node.pitch(alignment),
                            node.pixel_type(),
                            block_width,
                            block_height,
                        ),
                    };

                    let module = Module::from_ptx(&compile(f).unwrap()).unwrap();
                    let function = module.get_function("kernel").unwrap();

                    let dependencies = node
                        .dependencies()
                        .into_iter()
                        .map(|n| graph_nodes.get(&(n as *const _)).unwrap())
                        .chain(std::iter::once(&alloc_node))
                        .collect_vec();

                    graph
                        .add_kernel_node(
                            &dependencies,
                            &function,
                            block_width,
                            block_height,
                            160,
                            140,
                        )
                        .unwrap()
                }
            };

            assert!(device_ptrs.insert(node, device_ptr).is_none());
            assert!(graph_nodes.insert(node, graph_node).is_none());
        }

        let output_buffers = outputs
            .into_iter()
            .map(|(name, node)| {
                let (_node, buffer) = graph.add_mem_cpy_node(
                    &graph_nodes[&(node as *const _)],
                    MemCpyDirection::DeviceToHost,
                    node.width(),
                    node.height(),
                    node.pitch(alignment),
                    node.pixel_type().layout(),
                    device_ptrs[&(node as *const _)],
                )?;
                let buffer = Buffer {
                    inner: buffer,
                    height: node.height(),
                    width: node.width(),
                    pixel_type: node.pixel_type(),
                };
                Ok((name, buffer))
            })
            .try_collect()?;

        Ok(Self {
            input_buffers,
            output_buffers,
            executable_graph: graph.make_executable()?,
            stream: Stream::new(&cuda)?,
        })
    }

    pub fn call(
        &mut self,
        inputs: HashMap<String, DynamicImage>,
    ) -> cuda::Result<HashMap<String, DynamicImage>> {
        for (name, buffer) in self.input_buffers.iter_mut() {
            let image = inputs.get(name).unwrap();
            // Safe because the buffer is currently not being read from
            unsafe {
                buffer.copy_from_dynamic_image(image);
            }
        }

        // Safe if the kernels are written correctly
        unsafe { self.executable_graph.launch(&mut self.stream).unwrap() };
        self.stream.synchronize().unwrap();

        // Safe because the buffers are not being written to
        unsafe {
            Ok(self
                .output_buffers
                .iter()
                .map(|(name, buffer)| (name.clone(), buffer.to_dynamic_image()))
                .collect())
        }
    }
}
