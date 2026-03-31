use burn::prelude::*;
use nn::{
    Dropout, DropoutConfig, Linear, LinearConfig,
    pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig},
};

use super::{
    inception_a::InceptionAConfig, inception_aux::InceptionAuxConfig,
    inception_b::InceptionBConfig, inception_c::InceptionCConfig, inception_d::InceptionDConfig,
    inception_e::InceptionEConfig,
};

use super::{
    basic_block::{BasicConv2d, BasicConv2dConfig},
    inception_a::InceptionA,
    inception_aux::InceptionAux,
    inception_b::InceptionB,
    inception_c::InceptionC,
    inception_d::InceptionD,
    inception_e::InceptionE,
};

#[derive(Debug, Config)]
pub struct InceptionV3Config {
    pub conv2d_1a_3x3: BasicConv2dConfig,
    pub conv2d_2a_3x3: BasicConv2dConfig,
    pub conv2d_2b_3x3: BasicConv2dConfig,
    pub maxpool1: MaxPool2dConfig,
    pub conv2d_3b_1x1: BasicConv2dConfig,
    pub conv2d_4a_3x3: BasicConv2dConfig,
    pub maxpool2: MaxPool2dConfig,
    pub mixed_5b: InceptionAConfig,
    pub mixed_5c: InceptionAConfig,
    pub mixed_5d: InceptionAConfig,
    pub mixed_6a: InceptionBConfig,
    pub mixed_6b: InceptionCConfig,
    pub mixed_6c: InceptionCConfig,
    pub mixed_6d: InceptionCConfig,
    pub mixed_6e: InceptionCConfig,
    pub aux_logits: Option<InceptionAuxConfig>,
    pub mixed_7a: InceptionDConfig,
    pub mixed_7b: InceptionEConfig,
    pub mixed_7c: InceptionEConfig,
    pub avgpool: AdaptiveAvgPool2dConfig,
    pub dropout: DropoutConfig,
    pub fc: LinearConfig,
}

impl std::default::Default for InceptionV3Config {
    fn default() -> Self {
        Self::create(1000, true, 0.5)
    }
}

impl InceptionV3Config {
    pub fn create(num_classes: usize, aux_logits: bool, dropout: f64) -> Self {
        type ConvBlock = BasicConv2dConfig;

        let aux_logits = aux_logits;
        Self {
            conv2d_1a_3x3: ConvBlock::create(3, 32, 3).with_stride(2),
            conv2d_2a_3x3: ConvBlock::create(32, 32, 3),
            conv2d_2b_3x3: ConvBlock::create(32, 64, 3).with_padding(1),
            maxpool1: MaxPool2dConfig::new([3, 3]).with_strides([2, 2]),
            conv2d_3b_1x1: ConvBlock::create(64, 80, 1),
            conv2d_4a_3x3: ConvBlock::create(80, 192, 3),
            maxpool2: MaxPool2dConfig::new([3, 3]).with_strides([2, 2]),
            mixed_5b: InceptionAConfig::default_with_channels(192, 32),
            mixed_5c: InceptionAConfig::default_with_channels(256, 64),
            mixed_5d: InceptionAConfig::default_with_channels(288, 64),
            mixed_6a: InceptionBConfig::default_with_channels(288),
            mixed_6b: InceptionCConfig::default_with_channels(768, 128),
            mixed_6c: InceptionCConfig::default_with_channels(768, 160),
            mixed_6d: InceptionCConfig::default_with_channels(768, 160),
            mixed_6e: InceptionCConfig::default_with_channels(768, 192),
            aux_logits: if aux_logits {
                Some(InceptionAuxConfig::default_with_channels(768, num_classes))
            } else {
                None
            },
            mixed_7a: InceptionDConfig::default_with_channels(768),
            mixed_7b: InceptionEConfig::default_with_channels(1280),
            mixed_7c: InceptionEConfig::default_with_channels(2048),
            avgpool: AdaptiveAvgPool2dConfig::new([1, 1]),
            dropout: DropoutConfig::new(dropout),
            fc: LinearConfig::new(2048, num_classes),
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> InceptionV3<B> {
        InceptionV3 {
            conv2d_1a_3x3: self.conv2d_1a_3x3.init(device),
            conv2d_2a_3x3: self.conv2d_2a_3x3.init(device),
            conv2d_2b_3x3: self.conv2d_2b_3x3.init(device),
            maxpool1: self.maxpool1.init(),
            conv2d_3b_1x1: self.conv2d_3b_1x1.init(device),
            conv2d_4a_3x3: self.conv2d_4a_3x3.init(device),
            maxpool2: self.maxpool2.init(),
            mixed_5b: self.mixed_5b.init(device),
            mixed_5c: self.mixed_5c.init(device),
            mixed_5d: self.mixed_5d.init(device),
            mixed_6a: self.mixed_6a.init(device),
            mixed_6b: self.mixed_6b.init(device),
            mixed_6c: self.mixed_6c.init(device),
            mixed_6d: self.mixed_6d.init(device),
            mixed_6e: self.mixed_6e.init(device),
            aux_logits: self.aux_logits.as_ref().map(|item| item.init(device)),
            mixed_7a: self.mixed_7a.init(device),
            mixed_7b: self.mixed_7b.init(device),
            mixed_7c: self.mixed_7c.init(device),
            avgpool: self.avgpool.init(),
            dropout: self.dropout.init(),
            fc: self.fc.init(device),
        }
    }
}

#[derive(Debug, Module)]
pub struct InceptionV3<B: Backend> {
    pub conv2d_1a_3x3: BasicConv2d<B>,
    pub conv2d_2a_3x3: BasicConv2d<B>,
    pub conv2d_2b_3x3: BasicConv2d<B>,
    pub maxpool1: MaxPool2d,
    pub conv2d_3b_1x1: BasicConv2d<B>,
    pub conv2d_4a_3x3: BasicConv2d<B>,
    pub maxpool2: MaxPool2d,
    pub mixed_5b: InceptionA<B>,
    pub mixed_5c: InceptionA<B>,
    pub mixed_5d: InceptionA<B>,
    pub mixed_6a: InceptionB<B>,
    pub mixed_6b: InceptionC<B>,
    pub mixed_6c: InceptionC<B>,
    pub mixed_6d: InceptionC<B>,
    pub mixed_6e: InceptionC<B>,
    pub aux_logits: Option<InceptionAux<B>>,
    pub mixed_7a: InceptionD<B>,
    pub mixed_7b: InceptionE<B>,
    pub mixed_7c: InceptionE<B>,
    pub avgpool: AdaptiveAvgPool2d,
    pub dropout: Dropout,
    pub fc: Linear<B>,
}

impl<B: Backend> InceptionV3<B> {
    pub fn forward(&self, x: Tensor<B, 4>, aux: bool) -> (Tensor<B, 2>, Option<Tensor<B, 2>>) {
        let n = x.shape().dims[0];
        // N x 3 x 299 x 299
        debug_assert_eq!(x.shape().dims, [n, 3, 299, 299]);
        let x = self.conv2d_1a_3x3.forward(x);
        // n x 32 x 149 x 149
        debug_assert_eq!(x.shape().dims, [n, 32, 149, 149]);
        let x = self.conv2d_2a_3x3.forward(x);
        // n x 32 x 147 x 147
        debug_assert_eq!(x.shape().dims, [n, 32, 147, 147]);
        let x = self.conv2d_2b_3x3.forward(x);
        // n x 64 x 147 x 147
        debug_assert_eq!(x.shape().dims, [n, 64, 147, 147]);
        let x = self.maxpool1.forward(x);
        // n x 64 x 73 x 73
        debug_assert_eq!(x.shape().dims, [n, 64, 73, 73]);
        let x = self.conv2d_3b_1x1.forward(x);
        // n x 80 x 73 x 73
        debug_assert_eq!(x.shape().dims, [n, 80, 73, 73]);
        let x = self.conv2d_4a_3x3.forward(x);
        // n x 192 x 71 x 71
        debug_assert_eq!(x.shape().dims, [n, 192, 71, 71]);
        let x = self.maxpool2.forward(x);
        // n x 192 x 35 x 35
        debug_assert_eq!(x.shape().dims, [n, 192, 35, 35]);
        let x = self.mixed_5b.forward(x);
        // n x 256 x 35 x 35
        debug_assert_eq!(x.shape().dims, [n, 256, 35, 35]);
        let x = self.mixed_5c.forward(x);
        // n x 288 x 35 x 35
        debug_assert_eq!(x.shape().dims, [n, 288, 35, 35]);
        let x = self.mixed_5d.forward(x);
        // n x 288 x 35 x 35
        debug_assert_eq!(x.shape().dims, [n, 288, 35, 35]);
        let x = self.mixed_6a.forward(x);
        // n x 768 x 17 x 17
        debug_assert_eq!(x.shape().dims, [n, 768, 17, 17]);
        let x = self.mixed_6b.forward(x);
        // n x 768 x 17 x 17
        debug_assert_eq!(x.shape().dims, [n, 768, 17, 17]);
        let x = self.mixed_6c.forward(x);
        // n x 768 x 17 x 17
        debug_assert_eq!(x.shape().dims, [n, 768, 17, 17]);
        let x = self.mixed_6d.forward(x);
        // n x 768 x 17 x 17
        debug_assert_eq!(x.shape().dims, [n, 768, 17, 17]);
        let x = self.mixed_6e.forward(x);
        // N x 768 x 17 x 17
        let aux = if let (Some(aux_logits), true) = (&self.aux_logits, aux) {
            Some(aux_logits.forward(x.clone()))
        } else {
            None
        };
        // N x 768 x 17 x 17
        debug_assert_eq!(x.shape().dims, [n, 768, 17, 17]);
        let x = self.mixed_7a.forward(x);
        // N x 1280 x 8 x 8
        debug_assert_eq!(x.shape().dims, [n, 1280, 8, 8]);
        let x = self.mixed_7b.forward(x);
        // N x 2048 x 8 x 8
        debug_assert_eq!(x.shape().dims, [n, 2048, 8, 8]);
        let x = self.mixed_7c.forward(x);
        // N x 2048 x 8 x 8
        // Adaptive average pooling
        debug_assert_eq!(x.shape().dims, [n, 2048, 8, 8]);
        let x = self.avgpool.forward(x);
        // N x 2048 x 1 x 1
        debug_assert_eq!(x.shape().dims, [n, 2048, 1, 1]);
        let x = self.dropout.forward(x);
        // N x 2048 x 1 x 1
        debug_assert_eq!(x.shape().dims, [n, 2048, 1, 1]);
        let x = Tensor::flatten::<2>(x, 1, 3);
        // N x 2048
        debug_assert_eq!(x.shape().dims, [n, 2048]);
        let x = self.fc.forward(x);
        // N x 1000 (num_classes)
        debug_assert_eq!(x.shape().dims, [n, 1000]);
        (x, aux)
    }
}

#[cfg(test)]
mod test {
    use burn::{
        backend::{NdArray, ndarray::NdArrayDevice},
        prelude::*,
    };

    use super::InceptionV3Config;

    #[test]
    fn sanity_check() {
        type B = NdArray;
        let device = NdArrayDevice::default();
        let device = &device;

        let model = InceptionV3Config::default();
        let model = model.init::<B>(device);

        let input: Tensor<B, 4> = Tensor::ones([4, 3, 299, 299], device);
        let (output, _) = model.forward(input, false);
        assert_eq!(output.shape().dims, [4, 1000]);
    }
}
