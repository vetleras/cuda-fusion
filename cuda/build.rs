use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-changed=build.rs");

    let cuda_path = PathBuf::from(env!("CUDA_PATH"));
    let include_path = cuda_path.join("include");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    println!("cargo:rustc-link-lib=dylib=cuda");
    println!(
        "cargo:rustc-link-search=native={}",
        cuda_path.join("lib64").to_str().unwrap()
    );

    bindgen::builder()
        .header(include_path.join("cuda.h").to_str().unwrap())
        .allowlist_item("(i?)cu.*") // only allow items starting with "cu" (case insensitve)
        .new_type_alias("CUresult")
        .must_use_type("CUresult")
        .rustified_enum("CUmemAllocationType_enum")
        .rustified_enum("CUmemAllocationHandleType_enum")
        .rustified_enum("CUmemLocationType_enum")
        .rustified_enum("CUmemorytype_enum")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("binding generation failed")
        .write_to_file(out_path.join("driver_bindings.rs"))
        .expect("writing bindings failed");
}
