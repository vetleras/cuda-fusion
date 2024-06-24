use std::{fs, process};

const INTERFACE_RLIB: &'static [u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/libinterface.rlib"));

pub fn compile(item_fn: syn::ItemFn) -> Result<String, ()> {
    let panic_handler = panic_handler();
    let file: syn::File = syn::parse_quote! {
        #![no_std]
        #![feature(abi_ptx, stdsimd, asm_experimental_arch)]

        extern crate interface;

        use core::arch::nvptx::*;

        #[no_mangle]
        #item_fn

        #panic_handler
    };

    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("codegen.rs"), prettyplease::unparse(&file)).unwrap();
    fs::write(dir.path().join("libinterface.rlib"), INTERFACE_RLIB).unwrap();

    let output = process::Command::new("rustup")
        .stderr(process::Stdio::inherit())
        .current_dir(&dir)
        .arg("run")
        .arg("nightly-2022-10-13")
        .arg("rustc")
        .arg("codegen.rs")
        .arg("--target")
        .arg("nvptx64-nvidia-cuda")
        .arg("--crate-type=cdylib")
        .arg("-O")
        .arg("-C")
        .arg("lto=off")
        .arg("-L")
        .arg(".")
        .arg("-C")
        .arg("target-cpu=sm_75")
        .output()
        .unwrap();

    if output.status.success() {
        Ok(fs::read_to_string(dir.path().join("codegen.ptx")).unwrap())
    } else {
        Err(())
    }
}

// the panic handler is copied from code supplies by Muybridge. I'm not sure what the original source is
fn panic_handler() -> syn::ItemFn {
    syn::parse_quote! {
        #[panic_handler]
        fn panic(panic_info: &core::panic::PanicInfo) -> ! {
            unsafe {
                #[repr(C)]
                struct PanicPrintArgs {
                    file: *const u8,
                    line: u32,
                    tx: i32,
                    ty: i32,
                    tz: i32,
                    bx: i32,
                    by: i32,
                    bz: i32,
                }

                if let Some(location) = panic_info.location() {
                    let filename_len: usize = location.file().as_bytes().len();
                    let filename_buf: *mut u8 = core::arch::nvptx::malloc(filename_len + 1) as *mut u8;
                    core::ptr::copy(
                        location.file().as_bytes() as *const [u8] as *const u8,
                        filename_buf,
                        filename_len,
                    );
                    *filename_buf.add(filename_len) = b'\0';
                    let panic_print_args = PanicPrintArgs {
                        file: filename_buf as *const u8,
                        line: location.line(),
                        tx: core::arch::nvptx::_thread_idx_x(),
                        ty: core::arch::nvptx::_thread_idx_y(),
                        tz: core::arch::nvptx::_thread_idx_z(),
                        bx: core::arch::nvptx::_block_idx_x(),
                        by: core::arch::nvptx::_block_idx_y(),
                        bz: core::arch::nvptx::_block_idx_z(),
                    };
                    core::arch::nvptx::vprintf(
                        ("panic occured on cuda device in file `%s` at line `%u for thread (%u, %u, %u), block (%u, %u, %u)`\n\0").as_bytes().as_ptr(),
                        &panic_print_args as *const PanicPrintArgs as *const core::ffi::c_void
                    );
                } else {
                    core::arch::nvptx::vprintf(
                        ("panic occurred but can't get location information...\n\0")
                            .as_bytes()
                            .as_ptr(),
                        core::ptr::null(),
                    );
                }
                core::arch::nvptx::trap();
            }
        }
    }
}
