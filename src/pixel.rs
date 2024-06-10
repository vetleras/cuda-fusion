use std::alloc::Layout;

use interface::{Rgb, SharedMemory};
use quote::quote;
use quote::ToTokens;

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum PixelType {
    RgbU8,
    RgbF32,
}

impl PixelType {
    pub const fn layout(&self) -> Layout {
        match self {
            PixelType::RgbU8 => Layout::new::<Rgb<u8>>(),
            PixelType::RgbF32 => Layout::new::<Rgb<f32>>(),
        }
    }
}

impl ToTokens for PixelType {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        tokens.extend(match self {
            PixelType::RgbU8 => quote! {interface::Rgb<u8>},
            PixelType::RgbF32 => quote! {interface::Rgb<f32>},
        });
    }
}

pub unsafe trait Pixel: SharedMemory {
    type ImageCratePixel: image::Pixel;

    fn ty() -> PixelType;
    fn into_image_crate_pixel(&self) -> Self::ImageCratePixel;
    fn from_image_crate_pixel(pixel: Self::ImageCratePixel) -> Self;
}

unsafe impl Pixel for Rgb<u8> {
    type ImageCratePixel = image::Rgb<u8>;

    fn ty() -> PixelType {
        PixelType::RgbU8
    }

    fn into_image_crate_pixel(&self) -> Self::ImageCratePixel {
        image::Rgb([self.r, self.g, self.b])
    }

    fn from_image_crate_pixel(pixel: Self::ImageCratePixel) -> Self {
        let image::Rgb([r, g, b]) = pixel;
        Self { r, g, b }
    }
}

unsafe impl Pixel for Rgb<f32> {
    type ImageCratePixel = image::Rgb<f32>;

    fn ty() -> PixelType {
        PixelType::RgbF32
    }

    fn into_image_crate_pixel(&self) -> Self::ImageCratePixel {
        image::Rgb([self.r, self.g, self.b])
    }

    fn from_image_crate_pixel(pixel: Self::ImageCratePixel) -> Self {
        let image::Rgb([r, g, b]) = pixel;
        Self { r, g, b }
    }
}
