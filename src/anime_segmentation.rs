use candle_core::{Result, Tensor};
use candle_nn::{Activation, Module, VarBuilder};

use crate::isnet_dis::{ISNetDIS, ISNetGTEncoder};

pub struct AnimeSegmentation {
    net: ISNetDIS,

    #[allow(dead_code)]
    gt_encoder: ISNetGTEncoder,
}

impl AnimeSegmentation {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            net: ISNetDIS::load(vb.pp("net"), 3, 1)?,
            gt_encoder: ISNetGTEncoder::load(vb.pp("gt_encoder"), 1, 1)?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (x0, _) = self.net.forward(x)?;
        Activation::Sigmoid.forward(&x0[0])
    }
}
