pub mod cifar;
pub mod imagenet;
pub mod mnist;

use burn::prelude::*;

/// A single data sample for attack
#[derive(Clone, Debug)]
pub struct Sample<B: Backend> {
    pub image: Tensor<B, 4>,
    pub label: usize,
}

/// Dataset trait for loading samples
pub trait AttackDataset<B: Backend> {
    /// Get number of samples in dataset
    fn len(&self) -> usize;

    /// Get a single sample by index
    fn get(&self, index: usize) -> Option<Sample<B>>;
}
