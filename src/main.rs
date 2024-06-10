use std::{collections::HashMap, fs, path::Path};

use cuda::Cuda;
use cuda_fusion::{new_input, Transformation};
use interface::{Image, Patch, Rgb};
use macros::{map_image_kernel, map_patch_kernel, map_pixel_kernel};

#[map_pixel_kernel]
fn to_u8(px: Rgb<f32>) -> Rgb<u8> {
    px.into()
}

#[map_pixel_kernel]
fn to_f32(px: Rgb<u8>) -> Rgb<f32> {
    px.into()
}

#[map_patch_kernel]
fn convolve(patch: Patch<3, Rgb<f32>>) -> Rgb<f32> {
    let m = [[0.1, 0.2, 0.1], [-0.1, 0.5, -0.1], [0.1, 0.2, 0.1]];
    let mut px = Default::default();

    for r in 0..3 {
        for c in 0..3 {
            let w = m[r][c];
            px += patch.get(c, r) * w;
        }
    }
    px
}

#[map_image_kernel]
fn shroom_filter(img: Image<Rgb<u8>>, col: usize, row: usize) -> Rgb<u8> {
    interface::Rgb {
        r: row as u8,
        g: img.get(col, row).unwrap().g,
        b: col as u8,
    }
}

fn main() {
    // init cuda and describe transformation
    let cuda = Cuda::new().unwrap();

    let img = load_image("resources/kitchen.png");
    let (width, height) = (img.width() as usize, img.height() as usize);

    let a = new_input::<Rgb<u8>>("a".into(), width, height);
    let res = a.map_image(&shroom_filter, width, height);
    let res2 = a
        .map_pixel(&to_f32)
        .map_patch(&convolve)
        .map_pixel(&to_u8)
        .flip();
    let res3 = res.h_concat(&res2);
    let res4 = res.v_concat(&res2);

    let outputs = HashMap::from([
        ("res".into(), res.into_output()),
        ("res2".into(), res2.into_output()),
        ("res3".into(), res3.into_output()),
        ("res4".into(), res4.into_output()),
    ]);

    // compile and load transformation
    let mut t = Transformation::new(&cuda, outputs).unwrap();

    // give inputs and call transformation
    let input_imgs = HashMap::from([("a".into(), img)]);
    let output_imgs = t.call(input_imgs.clone()).unwrap();

    // write outputs
    for (name, img) in output_imgs {
        let filename = format!("{name}.png");
        println!(
            "delete image file {}: {:?}",
            &filename,
            fs::remove_file(&filename)
        );
        img.save(filename).unwrap();
    }
}

fn load_image<P: AsRef<Path>>(path: P) -> image::DynamicImage {
    image::DynamicImage::ImageRgb8(
        image::io::Reader::open(path)
            .unwrap()
            .decode()
            .unwrap()
            .to_rgb8(),
    )
}
