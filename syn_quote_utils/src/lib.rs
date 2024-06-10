use syn::{FnArg, Ident, ItemFn, Pat, ReturnType, Type, TypePath};

pub fn extract_inputs(f: &ItemFn) -> Vec<(&Ident, &TypePath)> {
    f.sig
        .inputs
        .iter()
        .map(|fn_arg| {
            let FnArg::Typed(pat_type) = fn_arg else {
                panic!();
            };
            let Pat::Ident(pat_ident) = &*pat_type.pat else {
                panic!();
            };
            let Type::Path(type_path) = &*pat_type.ty else {
                panic!();
            };
            (&pat_ident.ident, type_path)
        })
        .collect()
}

pub fn extract_output(f: &ItemFn) -> &TypePath {
    let ReturnType::Type(_, ref ty) = f.sig.output else {
        panic!();
    };

    match &**ty {
        Type::Path(type_path) => type_path,
        _ => panic!(),
    }
}
