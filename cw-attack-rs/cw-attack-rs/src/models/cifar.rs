use burn::{
    nn::{self, Linear, PaddingConfig2d, conv::Conv2d},
    prelude::*,
};

/// CIFAR-10 Model Architecture
///
/// Based on the Python implementation:
/// Conv2d(3, 64, 3, padding=1) -> ReLU -> MaxPool2d(2)
/// Conv2d(64, 64, 3, padding=1) -> ReLU -> MaxPool2d(2)
/// Conv2d(64, 128, 3, padding=1) -> ReLU
/// Conv2d(128, 128, 3, padding=1) -> ReLU -> MaxPool2d(2)
/// Flatten (128 * 4 * 4 = 2048)
/// Linear(2048, 256) -> ReLU -> Dropout(0.5)
/// Linear(256, 256) -> ReLU -> Dropout(0.5)
/// Linear(256, 10)
#[derive(Module, Debug)]
pub struct CifarModel<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    conv4: Conv2d<B>,
    pool: nn::pool::MaxPool2d,
    dropout: nn::Dropout,
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
    relu: nn::Relu,
}

impl<B: Backend> CifarModel<B> {
    pub fn new(device: &B::Device) -> Self {
        // Conv2d(3, 64, 3, padding=1)
        let conv1 = nn::conv::Conv2dConfig::new([3, 64], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        // Conv2d(64, 64, 3, padding=1)
        let conv2 = nn::conv::Conv2dConfig::new([64, 64], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        // Conv2d(64, 128, 3, padding=1)
        let conv3 = nn::conv::Conv2dConfig::new([64, 128], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        // Conv2d(128, 128, 3, padding=1)
        let conv4 = nn::conv::Conv2dConfig::new([128, 128], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        // MaxPool2d(2, 2)
        let pool = nn::pool::MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .init();

        // Dropout(0.5)
        let dropout = nn::DropoutConfig::new(0.5).init();

        // After conv layers: 32x32 -> 16x16 -> 8x8 -> 8x8 -> 8x8 -> 4x4
        // Flatten: 128 * 4 * 4 = 2048
        let fc1 = nn::LinearConfig::new(2048, 256).init(device);
        let fc2 = nn::LinearConfig::new(256, 256).init(device);
        let fc3 = nn::LinearConfig::new(256, 10).init(device);

        Self {
            conv1,
            conv2,
            conv3,
            conv4,
            pool,
            dropout,
            fc1,
            fc2,
            fc3,
            relu: nn::Relu::new(),
        }
    }

    /// Forward pass returning logits (no softmax)
    /// Input: [batch_size, 3, 32, 32]
    /// Output: [batch_size, 10]
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        // Conv1 + Pool: [B, 3, 32, 32] -> [B, 64, 32, 32] -> [B, 64, 16, 16]
        let x = self.conv1.forward(input);
        let x = self.relu.forward(x);
        let x = self.pool.forward(x);

        // Conv2 + Pool: [B, 64, 16, 16] -> [B, 64, 16, 16] -> [B, 64, 8, 8]
        let x = self.conv2.forward(x);
        let x = self.relu.forward(x);
        let x = self.pool.forward(x);

        // Conv3: [B, 64, 8, 8] -> [B, 128, 8, 8]
        let x = self.conv3.forward(x);
        let x = self.relu.forward(x);

        // Conv4 + Pool: [B, 128, 8, 8] -> [B, 128, 8, 8] -> [B, 128, 4, 4]
        let x = self.conv4.forward(x);
        let x = self.relu.forward(x);
        let x = self.pool.forward(x);

        // Flatten: [B, 128, 4, 4] -> [B, 2048]
        let [batch_size, channels, height, width] = x.dims();
        let x = x.reshape([batch_size, channels * height * width]);

        // FC1: [B, 2048] -> [B, 256]
        let x = self.fc1.forward(x);
        let x = self.relu.forward(x);
        let x = self.dropout.forward(x);

        // FC2: [B, 256] -> [B, 256]
        let x = self.fc2.forward(x);
        let x = self.relu.forward(x);
        let x = self.dropout.forward(x);

        // FC3: [B, 256] -> [B, 10]
        self.fc3.forward(x)
    }
}

impl<B: Backend> Default for CifarModel<B> {
    fn default() -> Self {
        Self::new(&B::Device::default())
    }
}
