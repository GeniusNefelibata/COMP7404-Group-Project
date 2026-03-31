use burn::{
    nn::{self, Linear, PaddingConfig2d, conv::Conv2d},
    prelude::*,
};

/// MNIST Model Architecture
///
/// Based on the Python implementation:
/// Conv2d(1, 32, 3, 1) -> ReLU
/// Conv2d(32, 64, 3, 1) -> ReLU
/// MaxPool2d(2) -> Dropout2d(0.25)
/// Flatten
/// Linear(9216, 128) -> ReLU -> Dropout2d(0.5)
/// Linear(128, 10)
#[derive(Module, Debug)]
pub struct MnistModel<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    pool: nn::pool::MaxPool2d,
    dropout1: nn::Dropout,
    dropout2: nn::Dropout,
    fc1: Linear<B>,
    fc2: Linear<B>,
    relu: nn::Relu,
}

impl<B: Backend> MnistModel<B> {
    pub fn new(device: &B::Device) -> Self {
        // Conv2d(1, 32, 3, 1) - no padding
        let conv1 = nn::conv::Conv2dConfig::new([1, 32], [3, 3])
            .with_padding(PaddingConfig2d::Valid)
            .init(device);

        // Conv2d(32, 64, 3, 1) - no padding
        let conv2 = nn::conv::Conv2dConfig::new([32, 64], [3, 3])
            .with_padding(PaddingConfig2d::Valid)
            .init(device);

        // MaxPool2d(2)
        let pool = nn::pool::MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .init();

        // Dropout layers
        let dropout1 = nn::DropoutConfig::new(0.25).init();
        let dropout2 = nn::DropoutConfig::new(0.5).init();

        // After conv layers: 28x28 -> 26x26 -> 24x24 -> 12x12 (after pool)
        // Flatten: 64 * 12 * 12 = 9216
        let fc1 = nn::LinearConfig::new(9216, 128).init(device);
        let fc2 = nn::LinearConfig::new(128, 10).init(device);

        Self {
            conv1,
            conv2,
            pool,
            dropout1,
            dropout2,
            fc1,
            fc2,
            relu: nn::Relu::new(),
        }
    }

    /// Forward pass returning logits (no softmax)
    /// Input: [batch_size, 1, 28, 28]
    /// Output: [batch_size, 10]
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        // Conv1: [B, 1, 28, 28] -> [B, 32, 26, 26]
        let x = self.conv1.forward(input);
        let x = self.relu.forward(x);

        // Conv2: [B, 32, 26, 26] -> [B, 64, 24, 24]
        let x = self.conv2.forward(x);
        let x = self.relu.forward(x);

        // MaxPool: [B, 64, 24, 24] -> [B, 64, 12, 12]
        let x = self.pool.forward(x);

        // Dropout1
        let x = self.dropout1.forward(x);

        // Flatten: [B, 64, 12, 12] -> [B, 9216]
        let [batch_size, channels, height, width] = x.dims();
        let x = x.reshape([batch_size, channels * height * width]);

        // FC1: [B, 9216] -> [B, 128]
        let x = self.fc1.forward(x);
        let x = self.relu.forward(x);

        // Dropout2
        let x = self.dropout2.forward(x);

        // FC2: [B, 128] -> [B, 10]
        self.fc2.forward(x)
    }
}

impl<B: Backend> Default for MnistModel<B> {
    fn default() -> Self {
        Self::new(&B::Device::default())
    }
}
