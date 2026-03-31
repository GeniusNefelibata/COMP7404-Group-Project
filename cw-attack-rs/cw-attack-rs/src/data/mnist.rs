use super::{AttackDataset, Sample};
use burn::prelude::*;
use burn::tensor::TensorData;
use std::fs::File;
use std::io::{BufReader, Read};

/// MNIST dataset wrapper for attack
pub struct MnistAttackDataset<B: Backend> {
    images: Vec<Vec<f32>>, // Normalized pixel values [0, 1]
    labels: Vec<u8>,
    device: B::Device,
}

impl<B: Backend> MnistAttackDataset<B> {
    /// Load MNIST test dataset from the standard binary files
    pub fn test(data_dir: &str, device: B::Device) -> anyhow::Result<Self> {
        let images_path = format!("{}/MNIST/raw/t10k-images-idx3-ubyte", data_dir);
        let labels_path = format!("{}/MNIST/raw/t10k-labels-idx1-ubyte", data_dir);

        let images = Self::load_images(&images_path)?;
        let labels = Self::load_labels(&labels_path)?;

        Ok(Self {
            images,
            labels,
            device,
        })
    }

    /// Load MNIST images from binary file
    fn load_images(path: &str) -> anyhow::Result<Vec<Vec<f32>>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read magic number (4 bytes)
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        let magic = u32::from_be_bytes(magic);
        if magic != 2051 {
            anyhow::bail!("Invalid MNIST image file: magic number {} != 2051", magic);
        }

        // Read number of images (4 bytes)
        let mut num_images = [0u8; 4];
        reader.read_exact(&mut num_images)?;
        let num_images = u32::from_be_bytes(num_images) as usize;

        // Read number of rows (4 bytes)
        let mut num_rows = [0u8; 4];
        reader.read_exact(&mut num_rows)?;
        let num_rows = u32::from_be_bytes(num_rows) as usize;

        // Read number of columns (4 bytes)
        let mut num_cols = [0u8; 4];
        reader.read_exact(&mut num_cols)?;
        let num_cols = u32::from_be_bytes(num_cols) as usize;

        // Read image data
        let mut images = Vec::with_capacity(num_images);
        for _ in 0..num_images {
            let mut image = vec![0u8; num_rows * num_cols];
            reader.read_exact(&mut image)?;
            // Normalize to [0, 1]
            let image: Vec<f32> = image.iter().map(|&p| p as f32 / 255.0).collect();
            images.push(image);
        }

        Ok(images)
    }

    /// Load MNIST labels from binary file
    fn load_labels(path: &str) -> anyhow::Result<Vec<u8>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read magic number (4 bytes)
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        let magic = u32::from_be_bytes(magic);
        if magic != 2049 {
            anyhow::bail!("Invalid MNIST label file: magic number {} != 2049", magic);
        }

        // Read number of labels (4 bytes)
        let mut num_labels = [0u8; 4];
        reader.read_exact(&mut num_labels)?;
        let num_labels = u32::from_be_bytes(num_labels) as usize;

        // Read label data
        let mut labels = vec![0u8; num_labels];
        reader.read_exact(&mut labels)?;

        Ok(labels)
    }

    fn prepare_item(&self, index: usize) -> Sample<B> {
        let image_data = &self.images[index];
        let label = self.labels[index] as usize;

        let data = TensorData::new(image_data.clone(), vec![1, 1, 28, 28]);
        let image = Tensor::<B, 4>::from_data(data.convert::<f32>(), &self.device);

        Sample { image, label }
    }
}

impl<B: Backend> AttackDataset<B> for MnistAttackDataset<B> {
    fn len(&self) -> usize {
        self.images.len()
    }

    fn get(&self, index: usize) -> Option<Sample<B>> {
        if index < self.len() {
            Some(self.prepare_item(index))
        } else {
            None
        }
    }
}

/// Load a single MNIST image from file
pub fn load_mnist_image<B: Backend>(
    path: &str,
    device: &B::Device,
) -> anyhow::Result<Tensor<B, 4>> {
    use image::ImageReader;

    let img = ImageReader::open(path)?.decode()?;
    let img = img.to_luma8();

    let (width, height) = (img.width() as usize, img.height() as usize);
    if width != 28 || height != 28 {
        anyhow::bail!("MNIST image must be 28x28, got {}x{}", width, height);
    }

    let pixels: Vec<f32> = img.pixels().map(|p| p[0] as f32 / 255.0).collect();

    let data = TensorData::new(pixels, vec![1, 1, 28, 28]);
    let tensor = Tensor::<B, 4>::from_data(data.convert::<f32>(), device);

    Ok(tensor)
}
