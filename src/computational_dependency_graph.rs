use std::{collections::HashSet, rc::Rc};

use crate::pixel::PixelType;

pub enum Node {
    Input {
        name: String,
        width: usize,
        height: usize,
        pixel_type: PixelType,
    },
    Operation(Operation),
}

pub enum Operation {
    MapPixel {
        dependency: Rc<Node>,
        f: syn::ItemFn,
        pixel_type: PixelType,
    },
    MapPatch {
        dependency: Rc<Node>,
        f: syn::ItemFn,
        dimension: usize,
        pixel_type: PixelType,
    },
    MapImage {
        dependency: Rc<Node>,
        f: syn::ItemFn,
        width: usize,
        height: usize,
        pixel_type: PixelType,
    },
    Flip {
        dependency: Rc<Node>,
    },
    HConcat {
        dependency_left: Rc<Node>,
        dependency_right: Rc<Node>,
    },
    VConcat {
        dependency_top: Rc<Node>,
        dependency_bottom: Rc<Node>,
    },
}

pub fn toposort(roots: Vec<&Node>) -> Vec<&Node> {
    let mut visited = HashSet::new();
    let mut result = Vec::new();
    let mut result_set = HashSet::new();
    let mut stack = roots;

    while let Some(node) = stack.pop() {
        let node_ptr = node as *const Node;
        if !result_set.contains(&node_ptr) {
            if visited.insert(node_ptr) {
                stack.push(node);
                stack.extend(node.dependencies())
            } else {
                result_set.insert(node_ptr);
                result.push(node);
            }
        }
    }
    result
}

impl Node {
    pub fn height(&self) -> usize {
        match self {
            Node::Input {
                name: _,
                height,
                width: _,
                pixel_type: _,
            } => *height,
            Node::Operation(o) => o.height(),
        }
    }

    pub fn width(&self) -> usize {
        match self {
            Node::Input {
                name: _,
                height: _,
                width,
                pixel_type: _,
            } => *width,
            Node::Operation(o) => o.width(),
        }
    }

    pub fn pixel_type(&self) -> PixelType {
        match self {
            Node::Input {
                name: _,
                height: _,
                width: _,
                pixel_type,
            } => *pixel_type,
            Node::Operation(o) => o.pixel_type(),
        }
    }

    pub fn pitch(&self, alignment: usize) -> usize {
        (self.width() * self.pixel_type().layout().size()).div_ceil(alignment) * alignment
    }

    pub fn dependencies(&self) -> Vec<&Self> {
        match self {
            Node::Input {
                name: _,
                height: _,
                width: _,
                pixel_type: _,
            } => Vec::new(),
            Node::Operation(o) => o.dependencies(),
        }
    }
}

impl Operation {
    fn height(&self) -> usize {
        match self {
            Operation::MapPixel {
                dependency: child,
                f: _,
                pixel_type: _,
            } => child.height(),

            Operation::MapPatch {
                dependency: child,
                f: _,
                dimension: _,
                pixel_type: _,
            } => child.height(),

            Operation::MapImage {
                dependency: _,
                f: _,
                height,
                width: _,
                pixel_type: _,
            } => *height,

            Operation::Flip { dependency } => dependency.height(),

            Operation::HConcat {
                dependency_left,
                dependency_right,
            } => {
                assert_eq!(dependency_left.height(), dependency_right.height());
                dependency_left.height()
            }

            Operation::VConcat {
                dependency_top,
                dependency_bottom,
            } => dependency_top.height() + dependency_bottom.height(),
        }
    }

    fn width(&self) -> usize {
        match self {
            Operation::MapPixel {
                dependency: child,
                f: _,
                pixel_type: _,
            } => child.width(),

            Operation::MapPatch {
                dependency: child,
                f: _,
                dimension: _,
                pixel_type: _,
            } => child.width(),

            Operation::MapImage {
                dependency: _,
                f: _,
                height: _,
                width,
                pixel_type: _,
            } => *width,

            Operation::Flip { dependency } => dependency.width(),

            Operation::HConcat {
                dependency_left,
                dependency_right,
            } => dependency_left.width() + dependency_right.width(),

            Operation::VConcat {
                dependency_top,
                dependency_bottom,
            } => {
                assert_eq!(dependency_top.width(), dependency_bottom.width());
                dependency_top.width()
            }
        }
    }

    fn pixel_type(&self) -> PixelType {
        match self {
            Operation::MapPixel {
                dependency: _,
                f: _,
                pixel_type,
            } => *pixel_type,

            Operation::MapPatch {
                dependency: _,
                f: _,
                dimension: _,
                pixel_type,
            } => *pixel_type,

            Operation::MapImage {
                dependency: _,
                f: _,
                height: _,
                width: _,
                pixel_type,
            } => *pixel_type,

            Operation::Flip { dependency } => dependency.pixel_type(),

            Operation::HConcat {
                dependency_left,
                dependency_right,
            } => {
                assert_eq!(dependency_left.pixel_type(), dependency_right.pixel_type());
                dependency_left.pixel_type()
            }

            Operation::VConcat {
                dependency_top,
                dependency_bottom,
            } => {
                assert_eq!(dependency_top.pixel_type(), dependency_bottom.pixel_type());
                dependency_top.pixel_type()
            }
        }
    }

    fn dependencies(&self) -> Vec<&Node> {
        match self {
            Operation::MapPixel {
                dependency,
                f: _,
                pixel_type: _,
            }
            | Operation::MapPatch {
                dependency,
                f: _,
                dimension: _,
                pixel_type: _,
            }
            | Operation::MapImage {
                dependency,
                f: _,
                height: _,
                width: _,
                pixel_type: _,
            }
            | Operation::Flip { dependency } => vec![&**dependency],

            Operation::HConcat {
                dependency_left,
                dependency_right,
            } => vec![dependency_left, dependency_right],

            Operation::VConcat {
                dependency_top,
                dependency_bottom,
            } => vec![dependency_top, dependency_bottom],
        }
    }
}
