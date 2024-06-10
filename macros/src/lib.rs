extern crate proc_macro;

use proc_macro::TokenStream;
use quote::{quote, ToTokens};
use syn::{parse_macro_input, parse_quote, FnArg, Ident, ItemFn, ReturnType, Type, TypePath};
use syn_quote_utils::{extract_inputs, extract_output};

#[proc_macro_attribute]
pub fn map_pixel_kernel(_args: TokenStream, item: TokenStream) -> TokenStream {
    let f = parse_macro_input!(item as ItemFn);

    let FnArg::Typed(fn_arg_type) = f.sig.inputs.first().unwrap() else {
        panic!();
    };

    let Type::Path(ref tp) = *fn_arg_type.ty else {
        panic!();
    };

    let a = tp.clone();

    let ReturnType::Type(_, ty) = f.sig.output.clone() else {
        panic!();
    };

    let Type::Path(ref tp) = *ty else {
        panic!();
    };

    let b = tp.path.clone();

    let name = f.sig.ident.clone();
    let function = f.into_token_stream().to_string();
    let src = function.as_str();

    quote! {
        #[allow(non_upper_case_globals)]
        const #name: ::kernel::MapPixelKernel<#a, #b> = ::kernel::MapPixelKernel::new(#src);
    }
    .into()
}

#[proc_macro_attribute]
pub fn map_patch_kernel(_args: TokenStream, item: TokenStream) -> TokenStream {
    let f = parse_macro_input!(item as ItemFn);

    let FnArg::Typed(fn_arg_type) = f.sig.inputs.first().unwrap() else {
        panic!();
    };

    let Type::Path(ref tp) = *fn_arg_type.ty else {
        panic!();
    };

    let a = tp.clone();

    let ReturnType::Type(_, ty) = f.sig.output.clone() else {
        panic!();
    };

    let Type::Path(ref tp) = *ty else {
        panic!();
    };

    let b = tp.path.clone();

    let name = f.sig.ident.clone();
    let function = f.into_token_stream().to_string();
    let src = function.as_str();

    quote! {
        #[allow(non_upper_case_globals)]
        const #name: ::kernel::MapPatchKernel<#a, #b> = ::kernel::MapPatchKernel::new(#src);
    }
    .into()
}

fn is_valid_map_image_kernel_input(ident: &Ident, type_path: &TypePath) -> bool {
    let usize_type_path: TypePath = parse_quote! {usize};
    &usize_type_path == type_path && ["col", "row"].contains(&ident.to_string().as_str())
}

#[proc_macro_attribute]
pub fn map_image_kernel(_args: TokenStream, item: TokenStream) -> TokenStream {
    let f = parse_macro_input!(item as ItemFn);

    let inputs = extract_inputs(&f);
    let mut input_iter = inputs.into_iter();

    let a = input_iter.next().unwrap().1.clone();
    let b = extract_output(&f).clone();

    for (ident, type_path) in input_iter {
        assert!(is_valid_map_image_kernel_input(ident, type_path));
    }

    let name = f.sig.ident.clone();
    let function = f.into_token_stream().to_string();
    let src = function.as_str();

    quote! {
        #[allow(non_upper_case_globals)]
        const #name: ::kernel::MapImageKernel<#a, #b> = ::kernel::MapImageKernel::new(#src);
    }
    .into()
}
