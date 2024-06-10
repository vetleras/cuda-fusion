use itertools::Itertools;
use syn::parse_quote;

use crate::pixel::PixelType;
use syn_quote_utils::extract_inputs;

pub fn map_pixel(
    ptr_in: usize,
    ptr_out: usize,
    width: usize,
    height: usize,
    pitch_in: usize,
    pitch_out: usize,
    pixel_type_in: PixelType,
    pixel_type_out: PixelType,
    f: &syn::ItemFn,
    block_width: usize,
    block_height: usize,
) -> syn::ItemFn {
    let inputs = extract_inputs(&f);
    let (ident, _type_path) = inputs.first().unwrap();

    let stmts = f.block.stmts.iter();

    parse_quote! {
        pub unsafe extern "ptx-kernel" fn kernel() {
            let col = _block_idx_x() as usize * #block_width + _thread_idx_x() as usize;
            let row = _block_idx_y() as usize * #block_height + _thread_idx_y() as usize;

            let img_in: interface::Image<#pixel_type_in> = interface::Image::new(
                #ptr_in as *mut u8, #width, #height, #pitch_in
            );

            let mut img_out: interface::Image<#pixel_type_out> = interface::Image::new(
                #ptr_out as *mut u8, #width, #height, #pitch_out
            );

            fn map_kernel(#ident: #pixel_type_in) -> #pixel_type_out {
                #(#stmts)*
            }

            if let Some(px) = img_in.get(col, row) {
                img_out[(col, row)] = map_kernel(px);
            }
        }
    }
}

pub fn map_patch(
    ptr_in: usize,
    ptr_out: usize,
    width: usize,
    height: usize,
    pitch_in: usize,
    pitch_out: usize,
    pixel_type_in: PixelType,
    pixel_type_out: PixelType,
    f: &syn::ItemFn,
    dimension: usize,
    block_width: usize,
    block_height: usize,
) -> syn::ItemFn {
    let inputs = extract_inputs(&f);
    let (ident, _type_path) = inputs.first().unwrap();

    let stmts = f.block.stmts.iter();

    assert!(dimension % 2 == 1);
    let padding = dimension / 2;

    let shared_memory_declearation = format!(
        ".shared .align {} .b8 SHARED[{}];",
        pixel_type_in.layout().align(),
        block_height * block_width * pixel_type_in.layout().size()
    );

    parse_quote! {
        pub unsafe extern "ptx-kernel" fn kernel() {
            let thread_col = _thread_idx_x() as usize;
            let thread_row = _thread_idx_y() as usize;
            let col = _block_idx_x() as usize * (#block_width - 2 * #padding) + thread_col - #padding;
            let row = _block_idx_y() as usize * (#block_width - 2 * #padding) + thread_row - #padding;

            let img_in: interface::Image<#pixel_type_in> = interface::Image::new(
                #ptr_in as *mut u8, #width, #height, #pitch_in
            );

            let px = img_in.get(col, row).unwrap_or_default();

            use interface::SharedMemory;
            core::arch::asm!(#shared_memory_declearation);
            let shared: *mut #pixel_type_in;
            core::arch::asm!("mov.u64 {}, SHARED;", out(reg64) shared);

            let thread_i = thread_col + thread_row * #block_width;
            px.store(shared.add(thread_i));

            if thread_col < #padding || thread_col + #padding >= #block_width ||
                thread_row < #padding || thread_row + #padding >= #block_height {
                return;
            }

            _syncthreads();

            let mut img_out: interface::Image<#pixel_type_out> = interface::Image::new(
                #ptr_out as *mut u8, #width, #height, #pitch_out
            );

            let patch: interface::Patch<#dimension, #pixel_type_in> = interface::Patch::new(
               shared, #block_width, thread_col, thread_row
            );

            fn map_kernel(#ident: interface::Patch<#dimension, #pixel_type_in>) -> #pixel_type_out {
                #(#stmts)*
            }

            if let Some(px) = img_out.get_mut(col, row) {
                *px = map_kernel(patch);
            }
        }
    }
}

pub fn map_image(
    ptr_in: usize,
    ptr_out: usize,
    width_in: usize,
    width_out: usize,
    height_in: usize,
    height_out: usize,
    pitch_in: usize,
    pitch_out: usize,
    pixel_type_in: PixelType,
    pixel_type_out: PixelType,
    f: &syn::ItemFn,
    block_width: usize,
    block_height: usize,
) -> syn::ItemFn {
    let stmts = f.block.stmts.iter();

    let fn_args = f.sig.inputs.iter().skip(1).collect_vec();

    let mut input_iter = extract_inputs(&f).into_iter();
    let (ident, _type_path) = input_iter.next().unwrap();
    let meta_args = input_iter.map(|(ident, _type_path)| ident).collect_vec();

    parse_quote! {
        pub unsafe extern "ptx-kernel" fn kernel() {
            let col = _block_idx_x() as usize * #block_width + _thread_idx_x() as usize;
            let row = _block_idx_y() as usize * #block_height + _thread_idx_y() as usize;

            let img_in: interface::Image<#pixel_type_in> = interface::Image::new(
                #ptr_in as *mut u8, #width_in, #height_in, #pitch_in
            );

            let mut img_out: interface::Image<#pixel_type_out> = interface::Image::new(
                #ptr_out as *mut u8, #width_out, #height_out, #pitch_out
            );

            fn map_kernel(#ident: interface::Image<#pixel_type_in>, #(#fn_args),*) -> #pixel_type_out {
                #(#stmts)*
            }

            if col < #width_out && row < #height_out {
                img_out[(col, row)] = map_kernel(img_in, #(#meta_args),*);
            }
        }
    }
}

pub fn flip(
    ptr_in: usize,
    ptr_out: usize,
    width: usize,
    height: usize,
    pitch: usize,
    pixel_type: PixelType,
    block_width: usize,
    block_height: usize,
) -> syn::ItemFn {
    parse_quote! {
        pub unsafe extern "ptx-kernel" fn kernel() {
            let col = _block_idx_x() as usize * #block_width + _thread_idx_x() as usize;
            let row = _block_idx_y() as usize * #block_height + _thread_idx_y() as usize;

            let img_in: interface::Image<#pixel_type> = interface::Image::new(
                #ptr_in as *mut u8, #width, #height, #pitch
            );

            let mut img_out: interface::Image<#pixel_type> = interface::Image::new(
                #ptr_out as *mut u8, #width, #height, #pitch
            );

            if let Some(px) = img_in.get(col, row) {
                img_out[(#width - col - 1, #height - row - 1)] = px;
            }
        }
    }
}

pub fn h_concat(
    ptr_left: usize,
    ptr_right: usize,
    ptr_out: usize,
    width_left: usize,
    width_right: usize,
    width_out: usize,
    height: usize,
    pitch_left: usize,
    pitch_right: usize,
    pitch_out: usize,
    pixel_type: PixelType,
    block_width: usize,
    block_height: usize,
) -> syn::ItemFn {
    parse_quote! {
        pub unsafe extern "ptx-kernel" fn kernel() {
            let col = _block_idx_x() as usize * #block_width + _thread_idx_x() as usize;
            let row = _block_idx_y() as usize * #block_height + _thread_idx_y() as usize;

            let img_left: interface::Image<#pixel_type> = interface::Image::new(
                #ptr_left as *mut u8, #width_left, #height, #pitch_left
            );

            let img_right: interface::Image<#pixel_type> = interface::Image::new(
                #ptr_right as *mut u8, #width_right, #height, #pitch_right
            );

            let mut img_out: interface::Image<#pixel_type> = interface::Image::new(
                #ptr_out as *mut u8, #width_out, #height, #pitch_out
            );

            if col < #width_left {
                if let Some(px) = img_left.get(col, row) {
                    img_out[(col, row)] = px;
                }
            } else if let Some(px) = img_right.get(col - #width_left, row) {
                img_out[(col, row)] = px;
            };
        }
    }
}

pub fn v_concat(
    ptr_top: usize,
    ptr_bottom: usize,
    ptr_out: usize,
    width: usize,
    height_top: usize,
    height_bottom: usize,
    height_out: usize,
    pitch: usize,
    pixel_type: PixelType,
    block_width: usize,
    block_height: usize,
) -> syn::ItemFn {
    parse_quote! {
        pub unsafe extern "ptx-kernel" fn kernel() {
            let col = _block_idx_x() as usize * #block_width + _thread_idx_x() as usize;
            let row = _block_idx_y() as usize * #block_height + _thread_idx_y() as usize;

            let img_top: interface::Image<#pixel_type> = interface::Image::new(
                #ptr_top as *mut u8, #width, #height_top, #pitch
            );

            let img_bottom: interface::Image<#pixel_type> = interface::Image::new(
                #ptr_bottom as *mut u8, #width, #height_bottom, #pitch
            );

            let mut img_out: interface::Image<#pixel_type> = interface::Image::new(
                #ptr_out as *mut u8, #width, #height_out, #pitch
            );

            if row < #height_top {
                if let Some(px) = img_top.get(col, row) {
                    img_out[(col, row)] = px;
                }
            } else if let Some(px) = img_bottom.get(col, row - #height_top) {
                img_out[(col, row)] = px;
            };
        }
    }
}
