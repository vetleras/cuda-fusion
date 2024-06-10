use std::{
    alloc::Layout,
    cell::UnsafeCell,
    collections::HashMap,
    ffi::{c_void, CString},
    fs,
    marker::PhantomData,
    mem::MaybeUninit,
    os::unix::ffi::OsStrExt,
    ptr,
    rc::Rc,
};

use itertools::Itertools;
use typed_arena::Arena;

use crate::{driver, module::Function, stream::Stream, Cuda, Result};

// The box should be unnecessary, but there is no way to allocate the [u8] with the required alignment with Rc directly
type Buffer = Rc<Box<UnsafeCell<[u8]>>>;

pub struct Graph<'a> {
    cuda: &'a Cuda,
    inner: driver::CUgraph,
    buffers: Arena<Buffer>,
}

pub struct Node<'a> {
    p: PhantomData<&'a Graph<'a>>,
    inner: driver::CUgraphNode,
}

#[derive(Clone, Copy, Debug)]
pub struct DevicePtr<'a> {
    p: PhantomData<&'a Graph<'a>>,
    inner: driver::CUdeviceptr,
}

impl<'a> DevicePtr<'a> {
    pub fn inner(&self) -> usize {
        self.inner as usize
    }
}

pub enum MemCpyDirection {
    HostToDevice,
    DeviceToHost,
}

impl<'a> Graph<'a> {
    pub fn new(cuda: &Cuda) -> Result<Graph> {
        let mut inner = ptr::null_mut();
        unsafe {
            driver::cuGraphCreate(&mut inner, 0).to_result()?;
        }
        Ok(Graph {
            inner,
            cuda,
            buffers: Arena::new(),
        })
    }

    pub fn get_dot(&self) -> Result<String> {
        let tempfile = tempfile::NamedTempFile::new().expect("should be able to create tempfile");
        let path = CString::new(tempfile.path().as_os_str().as_bytes()).unwrap();

        unsafe {
            driver::cuGraphDebugDotPrint(self.inner, path.as_ptr(), 0).to_result()?;
        };

        Ok(fs::read_to_string(tempfile.path()).expect("should be able to read file"))
    }

    pub fn add_mem_alloc_node(&self, height: usize, pitch: usize) -> Result<(Node, DevicePtr)> {
        let (node, device_ptr) = add_alloc_node_raw(self.inner, height, pitch)?;
        Ok((
            Node {
                p: PhantomData,
                inner: node,
            },
            DevicePtr {
                p: PhantomData,
                inner: device_ptr,
            },
        ))
    }

    pub fn add_mem_cpy_node(
        &self,
        dependency: &Node,
        direction: MemCpyDirection,
        width: usize,
        height: usize,
        pitch: usize,
        pixel_layout: Layout,
        device_ptr: DevicePtr,
    ) -> Result<(Node, Buffer)> {
        let size = width * height * pixel_layout.size();
        let layout = std::alloc::Layout::from_size_align(size, pixel_layout.align()).unwrap();
        let buffer = unsafe {
            Rc::new(Box::from_raw(UnsafeCell::from_mut(
                std::slice::from_raw_parts_mut(std::alloc::alloc(layout), size),
            )))
        };
        self.buffers.alloc(buffer.clone());

        let node = add_memcpy_node_raw(
            self.inner,
            dependency.inner,
            direction,
            buffer.get() as *mut u8,
            device_ptr.inner,
            width,
            height,
            pitch,
            pixel_layout.size(),
            self.cuda.context(),
        )?;
        Ok((
            Node {
                p: PhantomData,
                inner: node,
            },
            buffer,
        ))
    }

    pub fn add_kernel_node(
        &self,
        dependencies: &[&Node],
        function: &Function,
        block_width: usize,
        block_height: usize,
        grid_width: usize,
        grid_height: usize,
    ) -> driver::Result<Node> {
        let node = add_kernel_node_raw(
            self.inner,
            &dependencies.iter().map(|n| n.inner).collect_vec(),
            function.inner,
            block_width,
            block_height,
            grid_width,
            grid_height,
        )?;
        Ok(Node {
            p: PhantomData,
            inner: node,
        })
    }

    pub fn make_executable(mut self) -> driver::Result<ExecutableGraph<'a>> {
        add_free_nodes(self.inner)?;

        fs::write("cuda_graph.dot", self.get_dot()?).unwrap();

        let mut inner = ptr::null_mut();
        unsafe {
            driver::cuGraphInstantiateWithFlags(&mut inner, self.inner, 0).to_result()?;
        }

        Ok(ExecutableGraph {
            inner,
            p: PhantomData,
            _buffers: std::mem::take(&mut self.buffers),
        })
    }
}

impl<'a> Drop for Graph<'a> {
    fn drop(&mut self) {
        unsafe {
            driver::cuGraphDestroy(self.inner)
                .to_result()
                .expect("cuGraphDestroy failed while dropping Graph")
        }
    }
}

pub struct ExecutableGraph<'a> {
    p: PhantomData<&'a Cuda>,
    inner: driver::CUgraphExec,
    _buffers: Arena<Buffer>,
}

impl<'a> Drop for ExecutableGraph<'a> {
    fn drop(&mut self) {
        unsafe {
            driver::cuGraphExecDestroy(self.inner)
                .to_result()
                .expect("cuGraphExecDestroy failed while dropping ExecutableGraph")
        }
    }
}

impl<'a> ExecutableGraph<'a> {
    // Unsafety: It is the caller's responsibility to ensure that that all buffers returned from input and output node additions are handled appropriatly
    pub unsafe fn launch(&mut self, stream: &Stream) -> Result<()> {
        unsafe { driver::cuGraphLaunch(self.inner, stream.inner()).to_result() }
    }
}

// Returns dependency edges (from, to), where to has a dependency on from
fn get_dependency_edges(
    graph: driver::CUgraph,
) -> Result<Vec<(driver::CUgraphNode, driver::CUgraphNode)>> {
    unsafe {
        let mut edge_count = 0;
        driver::cuGraphGetEdges(graph, ptr::null_mut(), ptr::null_mut(), &mut edge_count)
            .to_result()?;

        let mut from = Vec::with_capacity(edge_count);
        let mut to = Vec::with_capacity(edge_count);
        driver::cuGraphGetEdges(graph, from.as_mut_ptr(), to.as_mut_ptr(), &mut edge_count)
            .to_result()?;
        from.set_len(edge_count);
        to.set_len(edge_count);

        Ok(std::iter::zip(from, to).collect())
    }
}

fn is_alloc_node(node: driver::CUgraphNode) -> Result<bool> {
    unsafe {
        let mut ty = 0;
        driver::cuGraphNodeGetType(node, &mut ty).to_result()?;
        Ok(ty == driver::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_MEM_ALLOC)
    }
}

fn get_device_ptr(node: driver::CUgraphNode) -> Result<driver::CUdeviceptr> {
    unsafe {
        let mut params = MaybeUninit::<driver::CUDA_MEM_ALLOC_NODE_PARAMS>::uninit();
        driver::cuGraphMemAllocNodeGetParams(node, params.as_mut_ptr()).to_result()?;
        Ok(params.assume_init().dptr)
    }
}

fn add_free_nodes(graph: driver::CUgraph) -> Result<()> {
    let mut dependency_map: HashMap<driver::CUgraphNode, Vec<_>> = HashMap::new();
    for (from, to) in get_dependency_edges(graph)? {
        if is_alloc_node(from)? {
            dependency_map.entry(from).or_default().push(to);
        }
    }

    for (alloc_node, dependents) in dependency_map {
        let device_ptr = get_device_ptr(alloc_node)?;
        unsafe {
            driver::cuGraphAddMemFreeNode(
                &mut ptr::null_mut(),
                graph,
                dependents.as_ptr(),
                dependents.len(),
                device_ptr,
            )
            .to_result()?;
        }
    }
    Ok(())
}

fn add_kernel_node_raw(
    graph: driver::CUgraph,
    dependencies: &[driver::CUgraphNode],
    function: driver::CUfunction,
    block_width: usize,
    block_height: usize,
    grid_width: usize,
    grid_height: usize,
) -> Result<driver::CUgraphNode> {
    let params = driver::CUDA_KERNEL_NODE_PARAMS {
        func: function,
        gridDimX: grid_width as u32,
        gridDimY: grid_height as u32,
        gridDimZ: 1,
        blockDimX: block_width as u32,
        blockDimY: block_height as u32,
        blockDimZ: 1,
        sharedMemBytes: 0,
        kernelParams: ptr::null_mut(),
        extra: ptr::null_mut(),
        kern: ptr::null_mut(),
        ctx: ptr::null_mut(),
    };

    let mut inner = ptr::null_mut();
    unsafe {
        driver::cuGraphAddKernelNode_v2(
            &mut inner,
            graph,
            dependencies.as_ptr(),
            dependencies.len(),
            &params,
        )
        .to_result()
        .map(|_| inner)
    }
}

fn add_alloc_node_raw(
    graph: driver::CUgraph,
    height: usize,
    pitch: usize,
) -> Result<(driver::CUgraphNode, driver::CUdeviceptr)> {
    let location = driver::CUmemLocation {
        type_: driver::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
        id: 0,
    };

    let props = driver::CUmemPoolProps {
        allocType: driver::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED,
        handleTypes: driver::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE,
        location,
        win32SecurityAttributes: ptr::null_mut(),
        maxSize: 0,
        reserved: [0; 56],
    };

    let mut params = driver::CUDA_MEM_ALLOC_NODE_PARAMS {
        poolProps: props,
        accessDescs: ptr::null(),
        accessDescCount: 0,
        bytesize: height * pitch,
        dptr: 0,
    };

    let mut inner = ptr::null_mut();
    unsafe {
        driver::cuGraphAddMemAllocNode(&mut inner, graph, ptr::null(), 0, &mut params)
            .to_result()
            .map(|_| (inner, params.dptr))
    }
}

fn add_memcpy_node_raw(
    graph: driver::CUgraph,
    dependency: driver::CUgraphNode,
    direction: MemCpyDirection,
    host_ptr: *mut u8,
    device_ptr: driver::CUdeviceptr,
    width: usize,
    height: usize,
    pitch: usize,
    pixel_size: usize,
    context: driver::CUcontext,
) -> Result<driver::CUgraphNode> {
    let host = (
        driver::CUmemorytype::CU_MEMORYTYPE_HOST,
        host_ptr,
        0,
        width * pixel_size,
    );

    let device = (
        driver::CUmemorytype::CU_MEMORYTYPE_DEVICE,
        ptr::null_mut(),
        device_ptr,
        pitch,
    );

    let (
        (src_memory_type, src_host, src_device_ptr, src_pitch),
        (dst_memory_type, dst_host, dst_device_ptr, dst_pitch),
    ) = match direction {
        MemCpyDirection::HostToDevice => (host, device),
        MemCpyDirection::DeviceToHost => (device, host),
    };

    let params = driver::CUDA_MEMCPY3D {
        srcMemoryType: src_memory_type,
        srcHost: src_host as *const c_void,
        srcDevice: src_device_ptr,
        srcPitch: src_pitch,
        srcHeight: height,
        dstMemoryType: dst_memory_type,
        dstHost: dst_host as *mut c_void,
        dstDevice: dst_device_ptr,
        dstPitch: dst_pitch,
        dstHeight: height,
        WidthInBytes: width * pixel_size,
        Height: height,
        Depth: 1,
        srcXInBytes: 0,
        srcY: 0,
        srcZ: 0,
        srcLOD: 0,
        srcArray: ptr::null_mut(),
        dstXInBytes: 0,
        dstY: 0,
        dstZ: 0,
        dstLOD: 0,
        dstArray: ptr::null_mut(),
        reserved0: ptr::null_mut(),
        reserved1: ptr::null_mut(),
    };

    let mut inner = ptr::null_mut();
    unsafe {
        driver::cuGraphAddMemcpyNode(&mut inner, graph, &dependency, 1, &params, context)
            .to_result()
            .map(|_| inner)
    }
}
