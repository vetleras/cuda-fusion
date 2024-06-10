use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=interface");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target_dir = out_dir.join("interface_rlib");
    let target = "nvptx64-nvidia-cuda";

    let output = std::process::Command::new("rustup")
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .env_remove("RUSTC")
        .env_remove("RUSTC_WRAPPER")
        .env_remove("RUSTC_WORKSPACE_WRAPPER")
        .env_remove("CARGO_FEATURE_HOST")
        .env("CARGO_CACHE_RUSTC_INFO", "0")
        .stderr(std::process::Stdio::inherit())
        .current_dir("interface")
        .arg("run")
        .arg("nightly-2022-10-13")
        .arg("cargo")
        .arg("rustc")
        .arg("--release")
        .arg("--target")
        .arg(target)
        .arg("--crate-type=rlib")
        .arg("--target-dir")
        .arg(&target_dir)
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "building interface rlib failed {output:?}"
    );

    let output_dir = target_dir.join(target).join("release");
    let dependencies =
        fs::read_to_string(output_dir.join("libinterface.d")).expect("no dependency file found");
    for dependency in dependencies.split_whitespace().skip(1) {
        println!("cargo:rerun-if-changed={}", dependency);
    }

    let rlib_name = "libinterface.rlib";
    fs::copy(output_dir.join(rlib_name), out_dir.join(rlib_name)).unwrap();
}
