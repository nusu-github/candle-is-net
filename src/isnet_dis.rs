use candle_core::{Result, Tensor};
use candle_nn::{
    Activation, batch_norm, BatchNorm, BatchNormConfig, conv2d, Conv2d, Conv2dConfig, Module,
    VarBuilder,
};

struct REBNCONV {
    conv_s1: Conv2d,
    bn_s1: BatchNorm,
    relu_s1: Activation,
}

impl REBNCONV {
    fn load(
        vb: VarBuilder,
        in_ch: usize,
        out_ch: usize,
        dirate: usize,
        stride: usize,
    ) -> Result<Self> {
        let conv_s1_config = Conv2dConfig {
            padding: 1 * dirate,
            dilation: 1 * dirate,
            stride,
            groups: 1,
        };

        let bn_s1_config = BatchNormConfig::default();

        Ok(Self {
            conv_s1: conv2d(in_ch, out_ch, 3, conv_s1_config, vb.pp("conv_s1"))?,
            bn_s1: batch_norm(out_ch, bn_s1_config, vb.pp("bn_s1"))?,
            relu_s1: Activation::Relu,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.relu_s1
            .forward(&self.bn_s1.forward_train(&self.conv_s1.forward(x)?)?)
    }
}

fn _upsample_like(x: &Tensor, y: &Tensor) -> Result<Tensor> {
    let (_, _, h, w) = y.dims4()?;
    x.interpolate2d(h, w) // TODO: bilinear補完実装待ち
}

struct RSU7 {
    rebnconvin: REBNCONV,

    rebnconv1: REBNCONV,
    rebnconv2: REBNCONV,
    rebnconv3: REBNCONV,
    rebnconv4: REBNCONV,
    rebnconv5: REBNCONV,
    rebnconv6: REBNCONV,
    rebnconv7: REBNCONV,

    rebnconv6d: REBNCONV,
    rebnconv5d: REBNCONV,
    rebnconv4d: REBNCONV,
    rebnconv3d: REBNCONV,
    rebnconv2d: REBNCONV,
    rebnconv1d: REBNCONV,
}

impl RSU7 {
    fn load(vb: VarBuilder, in_ch: usize, mid_ch: usize, out_ch: usize) -> Result<Self> {
        let rebnconvin = REBNCONV::load(vb.pp("rebnconvin"), in_ch, out_ch, 1, 1)?;

        let rebnconv1 = REBNCONV::load(vb.pp("rebnconv1"), out_ch, mid_ch, 1, 1)?;
        let rebnconv2 = REBNCONV::load(vb.pp("rebnconv2"), mid_ch, mid_ch, 1, 1)?;
        let rebnconv3 = REBNCONV::load(vb.pp("rebnconv3"), mid_ch, mid_ch, 1, 1)?;
        let rebnconv4 = REBNCONV::load(vb.pp("rebnconv4"), mid_ch, mid_ch, 1, 1)?;
        let rebnconv5 = REBNCONV::load(vb.pp("rebnconv5"), mid_ch, mid_ch, 1, 1)?;
        let rebnconv6 = REBNCONV::load(vb.pp("rebnconv6"), mid_ch, mid_ch, 1, 1)?;
        let rebnconv7 = REBNCONV::load(vb.pp("rebnconv7"), mid_ch, mid_ch, 2, 1)?;

        let rebnconv6d = REBNCONV::load(vb.pp("rebnconv6d"), mid_ch * 2, mid_ch, 1, 1)?;
        let rebnconv5d = REBNCONV::load(vb.pp("rebnconv5d"), mid_ch * 2, mid_ch, 1, 1)?;
        let rebnconv4d = REBNCONV::load(vb.pp("rebnconv4d"), mid_ch * 2, mid_ch, 1, 1)?;
        let rebnconv3d = REBNCONV::load(vb.pp("rebnconv3d"), mid_ch * 2, mid_ch, 1, 1)?;
        let rebnconv2d = REBNCONV::load(vb.pp("rebnconv2d"), mid_ch * 2, mid_ch, 1, 1)?;
        let rebnconv1d = REBNCONV::load(vb.pp("rebnconv1d"), mid_ch * 2, out_ch, 1, 1)?;

        Ok(Self {
            rebnconvin,
            rebnconv1,
            rebnconv2,
            rebnconv3,
            rebnconv4,
            rebnconv5,
            rebnconv6,
            rebnconv7,
            rebnconv6d,
            rebnconv5d,
            rebnconv4d,
            rebnconv3d,
            rebnconv2d,
            rebnconv1d,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hx = x;
        let hxin = self.rebnconvin.forward(hx)?;

        let hx1 = self.rebnconv1.forward(&hxin)?;
        let hx = hx1.max_pool2d_with_stride(2, 2)?;

        let hx2 = self.rebnconv2.forward(&hx)?;
        let hx = hx2.max_pool2d_with_stride(2, 2)?;

        let hx3 = self.rebnconv3.forward(&hx)?;
        let hx = hx3.max_pool2d_with_stride(2, 2)?;

        let hx4 = self.rebnconv4.forward(&hx)?;
        let hx = hx4.max_pool2d_with_stride(2, 2)?;

        let hx5 = self.rebnconv5.forward(&hx)?;
        let hx = hx5.max_pool2d_with_stride(2, 2)?;

        let hx6 = self.rebnconv6.forward(&hx)?;

        let hx7 = self.rebnconv7.forward(&hx6)?;

        let hx6d = self.rebnconv6d.forward(&Tensor::cat(&[hx7, hx6], 1)?)?;
        let hx6dup = _upsample_like(&hx6d, &hx5)?;

        let hx5d = self.rebnconv5d.forward(&Tensor::cat(&[hx6dup, hx5], 1)?)?;
        let hx5dup = _upsample_like(&hx5d, &hx4)?;

        let hx4d = self.rebnconv4d.forward(&Tensor::cat(&[hx5dup, hx4], 1)?)?;
        let hx4dup = _upsample_like(&hx4d, &hx3)?;

        let hx3d = self.rebnconv3d.forward(&Tensor::cat(&[hx4dup, hx3], 1)?)?;
        let hx3dup = _upsample_like(&hx3d, &hx2)?;

        let hx2d = self.rebnconv2d.forward(&Tensor::cat(&[hx3dup, hx2], 1)?)?;
        let hx2dup = _upsample_like(&hx2d, &hx1)?;

        let hx1d = self.rebnconv1d.forward(&Tensor::cat(&[hx2dup, hx1], 1)?)?;

        hx1d + hxin
    }
}

struct RSU6 {
    rebnconvin: REBNCONV,

    rebnconv1: REBNCONV,
    rebnconv2: REBNCONV,
    rebnconv3: REBNCONV,
    rebnconv4: REBNCONV,
    rebnconv5: REBNCONV,
    rebnconv6: REBNCONV,

    rebnconv5d: REBNCONV,
    rebnconv4d: REBNCONV,
    rebnconv3d: REBNCONV,
    rebnconv2d: REBNCONV,
    rebnconv1d: REBNCONV,
}

impl RSU6 {
    fn load(vb: VarBuilder, in_ch: usize, mid_ch: usize, out_ch: usize) -> Result<Self> {
        let rebnconvin = REBNCONV::load(vb.pp("rebnconvin"), in_ch, out_ch, 1, 1)?;

        let rebnconv1 = REBNCONV::load(vb.pp("rebnconv1"), out_ch, mid_ch, 1, 1)?;
        let rebnconv2 = REBNCONV::load(vb.pp("rebnconv2"), mid_ch, mid_ch, 1, 1)?;
        let rebnconv3 = REBNCONV::load(vb.pp("rebnconv3"), mid_ch, mid_ch, 1, 1)?;
        let rebnconv4 = REBNCONV::load(vb.pp("rebnconv4"), mid_ch, mid_ch, 1, 1)?;
        let rebnconv5 = REBNCONV::load(vb.pp("rebnconv5"), mid_ch, mid_ch, 1, 1)?;
        let rebnconv6 = REBNCONV::load(vb.pp("rebnconv6"), mid_ch, mid_ch, 2, 1)?;

        let rebnconv5d = REBNCONV::load(vb.pp("rebnconv5d"), mid_ch * 2, mid_ch, 1, 1)?;
        let rebnconv4d = REBNCONV::load(vb.pp("rebnconv4d"), mid_ch * 2, mid_ch, 1, 1)?;
        let rebnconv3d = REBNCONV::load(vb.pp("rebnconv3d"), mid_ch * 2, mid_ch, 1, 1)?;
        let rebnconv2d = REBNCONV::load(vb.pp("rebnconv2d"), mid_ch * 2, mid_ch, 1, 1)?;
        let rebnconv1d = REBNCONV::load(vb.pp("rebnconv1d"), mid_ch * 2, out_ch, 1, 1)?;
        Ok(Self {
            rebnconvin,
            rebnconv1,
            rebnconv2,
            rebnconv3,
            rebnconv4,
            rebnconv5,
            rebnconv6,
            rebnconv5d,
            rebnconv4d,
            rebnconv3d,
            rebnconv2d,
            rebnconv1d,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hx = x;

        let hxin = self.rebnconvin.forward(hx)?;

        let hx1 = self.rebnconv1.forward(&hxin)?;
        let hx = hx1.max_pool2d_with_stride(2, 2)?;

        let hx2 = self.rebnconv2.forward(&hx)?;
        let hx = hx2.max_pool2d_with_stride(2, 2)?;

        let hx3 = self.rebnconv3.forward(&hx)?;
        let hx = hx3.max_pool2d_with_stride(2, 2)?;

        let hx4 = self.rebnconv4.forward(&hx)?;
        let hx = hx4.max_pool2d_with_stride(2, 2)?;

        let hx5 = self.rebnconv5.forward(&hx)?;

        let hx6 = self.rebnconv6.forward(&hx5)?;

        let hx5d = self.rebnconv5d.forward(&Tensor::cat(&[hx6, hx5], 1)?)?;
        let hx5up = _upsample_like(&hx5d, &hx4)?;

        let hx4d = self.rebnconv4d.forward(&Tensor::cat(&[hx5up, hx4], 1)?)?;
        let hx4up = _upsample_like(&hx4d, &hx3)?;

        let hx3d = self.rebnconv3d.forward(&Tensor::cat(&[hx4up, hx3], 1)?)?;
        let hx3up = _upsample_like(&hx3d, &hx2)?;

        let hx2d = self.rebnconv2d.forward(&Tensor::cat(&[hx3up, hx2], 1)?)?;
        let hx2up = _upsample_like(&hx2d, &hx1)?;

        let hx1d = self.rebnconv1d.forward(&Tensor::cat(&[hx2up, hx1], 1)?)?;

        hx1d + hxin
    }
}

struct RSU5 {
    rebnconvin: REBNCONV,

    rebnconv1: REBNCONV,
    rebnconv2: REBNCONV,
    rebnconv3: REBNCONV,
    rebnconv4: REBNCONV,
    rebnconv5: REBNCONV,

    rebnconv4d: REBNCONV,
    rebnconv3d: REBNCONV,
    rebnconv2d: REBNCONV,
    rebnconv1d: REBNCONV,
}
impl RSU5 {
    fn load(vb: VarBuilder, in_ch: usize, mid_ch: usize, out_ch: usize) -> Result<Self> {
        let rebnconvin = REBNCONV::load(vb.pp("rebnconvin"), in_ch, out_ch, 1, 1)?;

        let rebnconv1 = REBNCONV::load(vb.pp("rebnconv1"), out_ch, mid_ch, 1, 1)?;
        let rebnconv2 = REBNCONV::load(vb.pp("rebnconv2"), mid_ch, mid_ch, 1, 1)?;
        let rebnconv3 = REBNCONV::load(vb.pp("rebnconv3"), mid_ch, mid_ch, 1, 1)?;
        let rebnconv4 = REBNCONV::load(vb.pp("rebnconv4"), mid_ch, mid_ch, 1, 1)?;
        let rebnconv5 = REBNCONV::load(vb.pp("rebnconv5"), mid_ch, mid_ch, 2, 1)?;

        let rebnconv4d = REBNCONV::load(vb.pp("rebnconv4d"), mid_ch * 2, mid_ch, 1, 1)?;
        let rebnconv3d = REBNCONV::load(vb.pp("rebnconv3d"), mid_ch * 2, mid_ch, 1, 1)?;
        let rebnconv2d = REBNCONV::load(vb.pp("rebnconv2d"), mid_ch * 2, mid_ch, 1, 1)?;
        let rebnconv1d = REBNCONV::load(vb.pp("rebnconv1d"), mid_ch * 2, out_ch, 1, 1)?;
        Ok(Self {
            rebnconvin,
            rebnconv1,
            rebnconv2,
            rebnconv3,
            rebnconv4,
            rebnconv5,
            rebnconv4d,
            rebnconv3d,
            rebnconv2d,
            rebnconv1d,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hx = x;

        let hxin = self.rebnconvin.forward(hx)?;

        let hx1 = self.rebnconv1.forward(&hxin)?;
        let hx = hx1.max_pool2d_with_stride(2, 2)?;

        let hx2 = self.rebnconv2.forward(&hx)?;
        let hx = hx2.max_pool2d_with_stride(2, 2)?;

        let hx3 = self.rebnconv3.forward(&hx)?;
        let hx = hx3.max_pool2d_with_stride(2, 2)?;

        let hx4 = self.rebnconv4.forward(&hx)?;

        let hx5 = self.rebnconv5.forward(&hx4)?;

        let hx4d = self.rebnconv4d.forward(&Tensor::cat(&[hx5, hx4], 1)?)?;
        let hx4up = _upsample_like(&hx4d, &hx3)?;

        let hx3d = self.rebnconv3d.forward(&Tensor::cat(&[hx4up, hx3], 1)?)?;
        let hx3up = _upsample_like(&hx3d, &hx2)?;

        let hx2d = self.rebnconv2d.forward(&Tensor::cat(&[hx3up, hx2], 1)?)?;
        let hx2up = _upsample_like(&hx2d, &hx1)?;

        let hx1d = self.rebnconv1d.forward(&Tensor::cat(&[hx2up, hx1], 1)?)?;

        hx1d + hxin
    }
}

struct RSU4 {
    rebnconvin: REBNCONV,

    rebnconv1: REBNCONV,
    rebnconv2: REBNCONV,
    rebnconv3: REBNCONV,
    rebnconv4: REBNCONV,

    rebnconv3d: REBNCONV,
    rebnconv2d: REBNCONV,
    rebnconv1d: REBNCONV,
}

impl RSU4 {
    fn load(vb: VarBuilder, in_ch: usize, mid_ch: usize, out_ch: usize) -> Result<Self> {
        let rebnconvin = REBNCONV::load(vb.pp("rebnconvin"), in_ch, out_ch, 1, 1)?;

        let rebnconv1 = REBNCONV::load(vb.pp("rebnconv1"), out_ch, mid_ch, 1, 1)?;
        let rebnconv2 = REBNCONV::load(vb.pp("rebnconv2"), mid_ch, mid_ch, 1, 1)?;
        let rebnconv3 = REBNCONV::load(vb.pp("rebnconv3"), mid_ch, mid_ch, 1, 1)?;
        let rebnconv4 = REBNCONV::load(vb.pp("rebnconv4"), mid_ch, mid_ch, 2, 1)?;

        let rebnconv3d = REBNCONV::load(vb.pp("rebnconv3d"), mid_ch * 2, mid_ch, 1, 1)?;
        let rebnconv2d = REBNCONV::load(vb.pp("rebnconv2d"), mid_ch * 2, mid_ch, 1, 1)?;
        let rebnconv1d = REBNCONV::load(vb.pp("rebnconv1d"), mid_ch * 2, out_ch, 1, 1)?;
        Ok(Self {
            rebnconvin,
            rebnconv1,
            rebnconv2,
            rebnconv3,
            rebnconv4,
            rebnconv3d,
            rebnconv2d,
            rebnconv1d,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hx = x;

        let hxin = self.rebnconvin.forward(hx)?;

        let hx1 = self.rebnconv1.forward(&hxin)?;
        let hx = hx1.max_pool2d_with_stride(2, 2)?;

        let hx2 = self.rebnconv2.forward(&hx)?;
        let hx = hx2.max_pool2d_with_stride(2, 2)?;

        let hx3 = self.rebnconv3.forward(&hx)?;

        let hx4 = self.rebnconv4.forward(&hx3)?;

        let hx3d = self.rebnconv3d.forward(&Tensor::cat(&[hx4, hx3], 1)?)?;
        let hx3up = _upsample_like(&hx3d, &hx2)?;

        let hx2d = self.rebnconv2d.forward(&Tensor::cat(&[hx3up, hx2], 1)?)?;
        let hx2up = _upsample_like(&hx2d, &hx1)?;

        let hx1d = self.rebnconv1d.forward(&Tensor::cat(&[hx2up, hx1], 1)?)?;

        hx1d + hxin
    }
}

struct RSU4F {
    rebnconvin: REBNCONV,

    rebnconv1: REBNCONV,
    rebnconv2: REBNCONV,
    rebnconv3: REBNCONV,
    rebnconv4: REBNCONV,

    rebnconv3d: REBNCONV,
    rebnconv2d: REBNCONV,
    rebnconv1d: REBNCONV,
}

impl RSU4F {
    fn load(vb: VarBuilder, in_ch: usize, mid_ch: usize, out_ch: usize) -> Result<Self> {
        let rebnconvin = REBNCONV::load(vb.pp("rebnconvin"), in_ch, out_ch, 1, 1)?;

        let rebnconv1 = REBNCONV::load(vb.pp("rebnconv1"), out_ch, mid_ch, 1, 1)?;
        let rebnconv2 = REBNCONV::load(vb.pp("rebnconv2"), mid_ch, mid_ch, 2, 1)?;
        let rebnconv3 = REBNCONV::load(vb.pp("rebnconv3"), mid_ch, mid_ch, 4, 1)?;

        let rebnconv4 = REBNCONV::load(vb.pp("rebnconv4"), mid_ch, mid_ch, 8, 1)?;

        let rebnconv3d = REBNCONV::load(vb.pp("rebnconv3d"), mid_ch * 2, mid_ch, 4, 1)?;
        let rebnconv2d = REBNCONV::load(vb.pp("rebnconv2d"), mid_ch * 2, mid_ch, 2, 1)?;
        let rebnconv1d = REBNCONV::load(vb.pp("rebnconv1d"), mid_ch * 2, out_ch, 1, 1)?;

        Ok(Self {
            rebnconvin,
            rebnconv1,
            rebnconv2,
            rebnconv3,
            rebnconv4,
            rebnconv3d,
            rebnconv2d,
            rebnconv1d,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hx = x;

        let hxin = self.rebnconvin.forward(hx)?;

        let hx1 = self.rebnconv1.forward(&hxin)?;
        let hx2 = self.rebnconv2.forward(&hx1)?;
        let hx3 = self.rebnconv3.forward(&hx2)?;

        let hx4 = self.rebnconv4.forward(&hx3)?;

        let hx3d = self.rebnconv3d.forward(&Tensor::cat(&[hx4, hx3], 1)?)?;
        let hx2d = self.rebnconv2d.forward(&Tensor::cat(&[hx3d, hx2], 1)?)?;
        let hx1d = self.rebnconv1d.forward(&Tensor::cat(&[hx2d, hx1], 1)?)?;

        hx1d + hxin
    }
}

#[allow(dead_code)]
struct MyREBNCONV {
    conv: Conv2d,
    bn: BatchNorm,
    rl: Activation,
}

impl MyREBNCONV {
    fn load(
        vb: VarBuilder,
        in_ch: usize,
        out_ch: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        let conv_config = Conv2dConfig {
            padding,
            dilation,
            stride,
            groups,
        };
        let conv = conv2d(in_ch, out_ch, kernel_size, conv_config, vb.pp("conv"))?;
        let bn = batch_norm(out_ch, BatchNormConfig::default(), vb.pp("bn"))?;
        Ok(Self {
            conv,
            bn,
            rl: Activation::Relu,
        })
    }

    #[allow(dead_code)]
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.rl
            .forward(&self.bn.forward_train(&self.conv.forward(x)?)?)
    }
}

#[allow(dead_code)]
pub struct ISNetGTEncoder {
    conv_in: MyREBNCONV,

    stage1: RSU7,
    stage2: RSU6,
    stage3: RSU5,
    stage4: RSU4,
    stage5: RSU4F,
    stage6: RSU4F,

    side1: Conv2d,
    side2: Conv2d,
    side3: Conv2d,
    side4: Conv2d,
    side5: Conv2d,
    side6: Conv2d,
}

impl ISNetGTEncoder {
    pub fn load(vb: VarBuilder, in_ch: usize, out_ch: usize) -> Result<Self> {
        let side_config = Conv2dConfig {
            padding: 1,
            dilation: 1,
            stride: 1,
            groups: 1,
        };

        Ok(Self {
            conv_in: MyREBNCONV::load(vb.pp("conv_in"), in_ch, 16, 3, 2, 1, 1, 1)?,

            stage1: RSU7::load(vb.pp("stage1"), 16, 16, 64)?,
            stage2: RSU6::load(vb.pp("stage2"), 64, 16, 64)?,
            stage3: RSU5::load(vb.pp("stage3"), 64, 32, 128)?,
            stage4: RSU4::load(vb.pp("stage4"), 128, 32, 256)?,
            stage5: RSU4F::load(vb.pp("stage5"), 256, 64, 512)?,
            stage6: RSU4F::load(vb.pp("stage6"), 512, 64, 512)?,

            side1: conv2d(64, out_ch, 3, side_config, vb.pp("side1"))?,
            side2: conv2d(64, out_ch, 3, side_config, vb.pp("side2"))?,
            side3: conv2d(128, out_ch, 3, side_config, vb.pp("side3"))?,
            side4: conv2d(256, out_ch, 3, side_config, vb.pp("side4"))?,
            side5: conv2d(512, out_ch, 3, side_config, vb.pp("side5"))?,
            side6: conv2d(512, out_ch, 3, side_config, vb.pp("side6"))?,
        })
    }

    #[allow(dead_code)]
    pub fn forward(&self, x: &Tensor) -> Result<([Tensor; 6], [Tensor; 6])> {
        let hx = x;

        let hxin = self.conv_in.forward(hx)?;

        // stage 1
        let hx1 = self.stage1.forward(&hxin)?;
        let hx = hxin.max_pool2d_with_stride(2, 2)?;

        // stage 2
        let hx2 = self.stage2.forward(&hx)?;
        let hx = hx2.max_pool2d_with_stride(2, 2)?;

        // stage 3
        let hx3 = self.stage3.forward(&hx)?;
        let hx = hx3.max_pool2d_with_stride(2, 2)?;

        // stage 4
        let hx4 = self.stage4.forward(&hx)?;
        let hx = hx4.max_pool2d_with_stride(2, 2)?;

        // stage 5
        let hx5 = self.stage5.forward(&hx)?;
        let hx = hx5.max_pool2d_with_stride(2, 2)?;

        // stage 6
        let hx6 = self.stage6.forward(&hx)?;

        // side output
        let d1 = self.side1.forward(&hx1)?;
        let d1 = _upsample_like(&d1, x)?;

        let d2 = self.side2.forward(&hx2)?;
        let d2 = _upsample_like(&d2, x)?;

        let d3 = self.side3.forward(&hx3)?;
        let d3 = _upsample_like(&d3, x)?;

        let d4 = self.side4.forward(&hx4)?;
        let d4 = _upsample_like(&d4, x)?;

        let d5 = self.side5.forward(&hx5)?;
        let d5 = _upsample_like(&d5, x)?;

        let d6 = self.side6.forward(&hx6)?;
        let d6 = _upsample_like(&d6, x)?;

        Ok(([d1, d2, d3, d4, d5, d6], [hx1, hx2, hx3, hx4, hx5, hx6]))
    }
}

pub struct ISNetDIS {
    conv_in: Conv2d,

    stage1: RSU7,
    stage2: RSU6,
    stage3: RSU5,
    stage4: RSU4,
    stage5: RSU4F,
    stage6: RSU4F,

    stage5d: RSU4F,
    stage4d: RSU4F,
    stage3d: RSU5,
    stage2d: RSU6,
    stage1d: RSU7,

    side1: Conv2d,
    side2: Conv2d,
    side3: Conv2d,
    side4: Conv2d,
    side5: Conv2d,
    side6: Conv2d,
}

impl ISNetDIS {
    pub fn load(vb: VarBuilder, in_ch: usize, out_ch: usize) -> Result<Self> {
        let conv_in_config = Conv2dConfig {
            padding: 1,
            dilation: 1,
            stride: 2,
            groups: 1,
        };
        let side_config = Conv2dConfig {
            padding: 1,
            dilation: 1,
            stride: 1,
            groups: 1,
        };

        Ok(Self {
            conv_in: conv2d(in_ch, 64, 3, conv_in_config, vb.pp("conv_in"))?,

            stage1: RSU7::load(vb.pp("stage1"), 64, 32, 64)?,
            stage2: RSU6::load(vb.pp("stage2"), 64, 32, 128)?,
            stage3: RSU5::load(vb.pp("stage3"), 128, 64, 256)?,
            stage4: RSU4::load(vb.pp("stage4"), 256, 128, 512)?,
            stage5: RSU4F::load(vb.pp("stage5"), 512, 256, 512)?,
            stage6: RSU4F::load(vb.pp("stage6"), 512, 256, 512)?,

            stage5d: RSU4F::load(vb.pp("stage5d"), 1024, 256, 512)?,
            stage4d: RSU4F::load(vb.pp("stage4d"), 1024, 128, 256)?,
            stage3d: RSU5::load(vb.pp("stage3d"), 512, 64, 128)?,
            stage2d: RSU6::load(vb.pp("stage2d"), 256, 32, 64)?,
            stage1d: RSU7::load(vb.pp("stage1d"), 128, 16, 64)?,

            side1: conv2d(64, out_ch, 3, side_config, vb.pp("side1"))?,
            side2: conv2d(64, out_ch, 3, side_config, vb.pp("side2"))?,
            side3: conv2d(128, out_ch, 3, side_config, vb.pp("side3"))?,
            side4: conv2d(256, out_ch, 3, side_config, vb.pp("side4"))?,
            side5: conv2d(512, out_ch, 3, side_config, vb.pp("side5"))?,
            side6: conv2d(512, out_ch, 3, side_config, vb.pp("side6"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<([Tensor; 6], [Tensor; 6])> {
        let hx = x;

        let hxin = self.conv_in.forward(hx)?;

        // stage 1
        let hx1 = self.stage1.forward(&hxin)?;
        let hx = hx1.max_pool2d_with_stride(2, 2)?;

        // stage 2
        let hx2 = self.stage2.forward(&hx)?;
        let hx = hx2.max_pool2d_with_stride(2, 2)?;

        // stage 3
        let hx3 = self.stage3.forward(&hx)?;
        let hx = hx3.max_pool2d_with_stride(2, 2)?;

        // stage 4
        let hx4 = self.stage4.forward(&hx)?;
        let hx = hx4.max_pool2d_with_stride(2, 2)?;

        // stage 5
        let hx5 = self.stage5.forward(&hx)?;
        let hx = hx5.max_pool2d_with_stride(2, 2)?;

        // stage 6
        let hx6 = self.stage6.forward(&hx)?;
        let hx6up = _upsample_like(&hx6, &hx5)?;

        // Decoder
        let hx5d = self.stage5d.forward(&Tensor::cat(&[&hx6up, &hx5], 1)?)?;
        let hx5dup = _upsample_like(&hx5d, &hx4)?;

        let hx4d = self.stage4d.forward(&Tensor::cat(&[&hx5dup, &hx4], 1)?)?;
        let hx4dup = _upsample_like(&hx4d, &hx3)?;

        let hx3d = self.stage3d.forward(&Tensor::cat(&[&hx4dup, &hx3], 1)?)?;
        let hx3dup = _upsample_like(&hx3d, &hx2)?;

        let hx2d = self.stage2d.forward(&Tensor::cat(&[&hx3dup, &hx2], 1)?)?;
        let hx2dup = _upsample_like(&hx2d, &hx1)?;

        let hx1d = self.stage1d.forward(&Tensor::cat(&[&hx2dup, &hx1], 1)?)?;

        // side output
        let d1 = self.side1.forward(&hx1d)?;
        let d1 = _upsample_like(&d1, x)?;

        let d2 = self.side2.forward(&hx2d)?;
        let d2 = _upsample_like(&d2, x)?;

        let d3 = self.side3.forward(&hx3d)?;
        let d3 = _upsample_like(&d3, x)?;

        let d4 = self.side4.forward(&hx4d)?;
        let d4 = _upsample_like(&d4, x)?;

        let d5 = self.side5.forward(&hx5d)?;
        let d5 = _upsample_like(&d5, x)?;

        let d6 = self.side6.forward(&hx6)?;
        let d6 = _upsample_like(&d6, x)?;

        Ok((
            [d1, d2, d3, d4, d5, d6],
            [hx1d, hx2d, hx3d, hx4d, hx5d, hx6],
        ))
    }
}
