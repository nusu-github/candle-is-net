use candle_core::{Device, DType, IndexOp, Result, Tensor};
use candle_nn::VarBuilder;
use image::{DynamicImage, ImageBuffer, imageops, Luma, RgbImage};
use imageops::FilterType;

use crate::model::Model;

mod model;
mod isnet_dis;

type Gray32FImage = ImageBuffer<Luma<f32>, Vec<f32>>;

type Crop = [u32; 4];

const IMAGE_SIZE: u32 = 1024;

fn preprocess(image: &RgbImage) -> anyhow::Result<(Tensor, Crop)> {
    let image = imageops::resize(image, IMAGE_SIZE, IMAGE_SIZE, FilterType::Lanczos3);
    let (width, height) = image.dimensions();
    let mut img_buf = RgbImage::new(IMAGE_SIZE, IMAGE_SIZE);
    let x1 = (IMAGE_SIZE - width) / 2;
    let y1 = (IMAGE_SIZE - height) / 2;
    let x2 = x1 + width;
    let y2 = y1 + height;
    imageops::overlay(&mut img_buf, &image, x1.into(), y1.into());

    let tensor = Tensor::from_vec(
        DynamicImage::from(img_buf).into_rgb32f().into_raw(),
        (IMAGE_SIZE as usize, IMAGE_SIZE as usize, 3),
        &Device::Cpu,
    )?
    .to_dtype(DType::F32)?
    .permute((2, 0, 1))?
    .transpose(0, 0)?
    .unsqueeze(0)?;

    Ok((tensor, [x1, y1, x2, y2]))
}

fn apply_mask(mut image: RgbImage, mask: &Gray32FImage) -> RgbImage {
    image
        .pixels_mut()
        .zip(mask.pixels())
        .for_each(|(out_pixel, mask_pixel)| {
            let [r, g, b] = out_pixel.0;
            let a = mask_pixel[0];
            let r = (a * (r as f32)) as u8;
            let g = (a * (g as f32)) as u8;
            let b = (a * (b as f32)) as u8;
            out_pixel.0 = [r, g, b];
        });
    image
}

fn task(tensor: Tensor, crop: Crop) -> Result<Tensor> {
    let device = Device::Cpu;
    let dtype = DType::F32;

    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&["isnetis.safetensors"], dtype, &device) }?;

    let model = Model::load(vb)?;

    let tensor = model.forward(&tensor.to_device(&device)?)?;

    let [x1, y1, x2, y2] = crop;
    let x1 = x1 as usize;
    let y1 = y1 as usize;
    let x2 = x2 as usize;
    let y2 = y2 as usize;

    tensor
        .squeeze(0)?
        .permute((1, 2, 0))?
        .i((x1..x2, y1..y2, ..))
}

fn main() -> anyhow::Result<()> {
    let image = image::open("test.jpg")?.into_rgb8();
    let (tensor, crop) = preprocess(&image)?;

    let tensor = task(tensor, crop)?;

    let (w, h, _) = tensor.dims3()?;
    let mask = Gray32FImage::from_raw(w as u32, h as u32, tensor.flatten_all()?.to_vec1::<f32>()?)
        .unwrap();
    let (w, h) = image.dimensions();
    let mask = imageops::resize(&mask, w, h, FilterType::Lanczos3);
    let image = apply_mask(image, &mask);

    image.save("out.png")?;
    DynamicImage::from(mask).into_luma8().save("mask.png")?;

    Ok(())
}
