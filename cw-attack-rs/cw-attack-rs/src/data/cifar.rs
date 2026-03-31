use super::{AttackDataset, Sample};
use burn::prelude::*;
use std::fs;
use std::path::Path;

/// CIFAR-10 dataset wrapper for attack
pub struct CifarAttackDataset<B: Backend> {
    images: Vec<Vec<u8>>, // Raw image data (3 * 32 * 32 = 3072 bytes each)
    labels: Vec<u8>,
    device: B::Device,
}

impl<B: Backend> CifarAttackDataset<B> {
    pub fn test(data_dir: &str, device: B::Device) -> anyhow::Result<Self> {
        let test_batch_path = Path::new(data_dir).join("cifar-10-batches-py/test_batch");

        if !test_batch_path.exists() {
            anyhow::bail!("CIFAR-10 test batch not found at {:?}", test_batch_path);
        }

        let data = fs::read(&test_batch_path)?;
        let (images, labels) = Self::parse_batch(&data)?;

        Ok(Self {
            images,
            labels,
            device,
        })
    }

    fn parse_batch(data: &[u8]) -> anyhow::Result<(Vec<Vec<u8>>, Vec<u8>)> {
        // CIFAR-10 binary format: <1 byte label><3072 bytes image>...
        let mut images = Vec::new();
        let mut labels = Vec::new();

        let record_size = 1 + 3072; // 1 byte label + 3072 bytes image
        let num_records = data.len() / record_size;

        for i in 0..num_records {
            let offset = i * record_size;
            let label = data[offset];
            let image_data = data[offset + 1..offset + record_size].to_vec();

            labels.push(label);
            images.push(image_data);
        }

        Ok((images, labels))
    }

    fn prepare_item(&self, index: usize) -> Sample<B> {
        let image_data = &self.images[index];
        let label = self.labels[index] as usize;

        // CIFAR-10 stores images as [R, G, B] for each pixel, row-major order
        // Convert to [3, 32, 32] format
        let mut pixels: Vec<f32> = Vec::with_capacity(3 * 32 * 32);

        for c in 0..3 {
            for i in 0..1024 {
                pixels.push(image_data[c * 1024 + i] as f32 / 255.0);
            }
        }

        let data = TensorData::new(pixels, vec![3, 32, 32]);
        let tensor = Tensor::<B, 3>::from_data(data.convert::<f32>(), &self.device);

        // Add batch dimension: [1, 3, 32, 32]
        let image = tensor.unsqueeze::<4>();

        Sample { image, label }
    }
}

impl<B: Backend> AttackDataset<B> for CifarAttackDataset<B> {
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

/// Load a single CIFAR image from file
pub fn load_cifar_image<B: Backend>(
    path: &str,
    device: &B::Device,
) -> anyhow::Result<Tensor<B, 4>> {
    use image::ImageReader;

    let img = ImageReader::open(path)?.decode()?;
    let img = img.to_rgb8();

    let (width, height) = (img.width() as usize, img.height() as usize);
    if width != 32 || height != 32 {
        anyhow::bail!("CIFAR image must be 32x32, got {}x{}", width, height);
    }

    let pixels: Vec<f32> = img
        .pixels()
        .flat_map(|p| {
            vec![
                p[0] as f32 / 255.0,
                p[1] as f32 / 255.0,
                p[2] as f32 / 255.0,
            ]
        })
        .collect();

    let data = TensorData::new(pixels, vec![1, 3, 32, 32]);
    let tensor = Tensor::<B, 4>::from_data(data.convert::<f32>(), device);

    Ok(tensor)
}
