use super::{AttackDataset, Sample};
use burn::prelude::*;
use std::fs;
use std::path::Path;

/// ImageNet dataset wrapper for attack
/// Loads sample images from a directory
pub struct ImageNetAttackDataset<B: Backend> {
    image_paths: Vec<std::path::PathBuf>,
    labels: Vec<usize>, // Will be 0 for unknown labels
    device: B::Device,
}

impl<B: Backend> ImageNetAttackDataset<B> {
    pub fn test(data_dir: &str, device: B::Device) -> anyhow::Result<Self> {
        let imagenet_dir = Path::new(data_dir).join("imagenet-sample-images");

        if !imagenet_dir.exists() {
            anyhow::bail!("ImageNet sample images not found at {:?}", imagenet_dir);
        }

        // Read all JPEG files from the directory
        let mut image_paths = Vec::new();
        let mut labels = Vec::new();

        for entry in fs::read_dir(&imagenet_dir)? {
            let entry = entry?;
            let path = entry.path();

            if let Some(ext) = path.extension() {
                if ext.eq_ignore_ascii_case("jpeg") || ext.eq_ignore_ascii_case("jpg") {
                    // Try to extract class index from filename
                    // Format: nXXXXXXXX_classname.JPEG
                    let label = Self::extract_label_from_filename(&path);
                    image_paths.push(path);
                    labels.push(label);
                }
            }
        }

        // Sort by filename for consistent ordering
        let mut pairs: Vec<_> = image_paths.into_iter().zip(labels.into_iter()).collect();
        pairs.sort_by(|a, b| a.0.file_name().cmp(&b.0.file_name()));

        let (image_paths, labels): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();

        println!("Loaded {} ImageNet sample images", image_paths.len());

        Ok(Self {
            image_paths,
            labels,
            device,
        })
    }

    /// Extract label from ImageNet filename
    /// Format: n01440764_tench.JPEG -> extract synset ID
    fn extract_label_from_filename(path: &Path) -> usize {
        if let Some(filename) = path.file_stem() {
            let name = filename.to_string_lossy();
            // Try to parse the synset ID (first 9 characters after 'n')
            if name.len() >= 10 && name.starts_with('n') {
                if let Ok(id) = name[1..10].parse::<usize>() {
                    // Map synset ID to a simpler index (0-999)
                    // This is a simplified mapping - in practice you'd use the official synset mapping
                    return id % 1000;
                }
            }
        }
        0 // Default to 0 if we can't parse
    }

    fn prepare_item(&self, index: usize) -> Sample<B> {
        let image_path = &self.image_paths[index];
        let label = self.labels[index];

        // Load and preprocess image
        let image = load_imagenet_image::<B>(image_path.to_str().unwrap(), &self.device)
            .expect("Failed to load ImageNet image");

        Sample { image, label }
    }
}

impl<B: Backend> AttackDataset<B> for ImageNetAttackDataset<B> {
    fn len(&self) -> usize {
        self.image_paths.len()
    }

    fn get(&self, index: usize) -> Option<Sample<B>> {
        if index < self.len() {
            Some(self.prepare_item(index))
        } else {
            None
        }
    }
}

/// Load a single ImageNet image from file
/// Resizes to 299x299 (Inception V3 input size) and normalizes
pub fn load_imagenet_image<B: Backend>(
    path: &str,
    device: &B::Device,
) -> anyhow::Result<Tensor<B, 4>> {
    use image::ImageReader;

    let img = ImageReader::open(path)?.decode()?;

    // Resize to 299x299 (Inception V3 input size)
    let img = img.resize_exact(299, 299, image::imageops::FilterType::Lanczos3);
    let img = img.to_rgb8();

    // Convert to tensor and normalize
    // ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];

    let mut pixels: Vec<f32> = Vec::with_capacity(3 * 299 * 299);

    for c in 0..3 {
        for y in 0..299 {
            for x in 0..299 {
                let pixel = img.get_pixel(x, y)[c] as f32 / 255.0;
                let normalized = (pixel - mean[c]) / std[c];
                pixels.push(normalized);
            }
        }
    }

    let data = TensorData::new(pixels, vec![1, 3, 299, 299]);
    let tensor = Tensor::<B, 4>::from_data(data.convert::<f32>(), device);

    Ok(tensor)
}
