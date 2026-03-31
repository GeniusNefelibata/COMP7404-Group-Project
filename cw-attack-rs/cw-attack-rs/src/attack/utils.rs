use super::AttackResult;
use burn::prelude::*;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Metrics computed from attack results
#[derive(Debug, Default)]
pub struct AttackMetrics {
    pub success_rate: f64,
    pub mean_l2: f64,
    pub median_l2: f64,
    pub std_l2: f64,
    pub min_l2: f64,
    pub max_l2: f64,
    pub total_samples: usize,
    pub successful_samples: usize,
    pub total_time: f64,
    pub mean_time: f64,
}

impl AttackMetrics {
    /// Compute metrics from attack results
    pub fn from_results(results: &[AttackResult]) -> Self {
        let total_samples = results.len();
        let successful: Vec<&AttackResult> = results.iter().filter(|r| r.success).collect();
        let successful_samples = successful.len();

        if successful.is_empty() {
            let total_time: f64 = results.iter().map(|r| r.time_seconds).sum();
            return Self {
                success_rate: 0.0,
                mean_l2: f64::INFINITY,
                median_l2: f64::INFINITY,
                std_l2: f64::INFINITY,
                min_l2: f64::INFINITY,
                max_l2: f64::INFINITY,
                total_samples,
                successful_samples: 0,
                total_time,
                mean_time: total_time / total_samples as f64,
            };
        }

        let l2_distances: Vec<f64> = successful.iter().map(|r| r.l2_distance).collect();
        let total_time: f64 = results.iter().map(|r| r.time_seconds).sum();

        let mean_l2 = l2_distances.iter().sum::<f64>() / l2_distances.len() as f64;
        let min_l2 = l2_distances.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_l2 = l2_distances
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // Compute median
        let mut sorted = l2_distances.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_l2 = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        // Compute std
        let variance = l2_distances
            .iter()
            .map(|&x| (x - mean_l2).powi(2))
            .sum::<f64>()
            / l2_distances.len() as f64;
        let std_l2 = variance.sqrt();

        Self {
            success_rate: (successful_samples as f64 / total_samples as f64) * 100.0,
            mean_l2,
            median_l2,
            std_l2,
            min_l2,
            max_l2,
            total_samples,
            successful_samples,
            total_time,
            mean_time: total_time / total_samples as f64,
        }
    }

    /// Format as human-readable text report
    pub fn to_text_report(&self) -> String {
        format!(
            r#"--- Attack Metrics ---
Success Rate: {:.2}%
Total Samples: {}
Successful: {}
Failed: {}

--- L2 Distance Statistics ---
Mean: {:.4}
Median: {:.4}
Std Dev: {:.4}
Min: {:.4}
Max: {:.4}

--- Time Statistics ---
Total Time: {:.2}s
Mean Time per Sample: {:.3}s
"#,
            self.success_rate,
            self.total_samples,
            self.successful_samples,
            self.total_samples - self.successful_samples,
            self.mean_l2,
            self.median_l2,
            self.std_l2,
            self.min_l2,
            self.max_l2,
            self.total_time,
            self.mean_time,
        )
    }
}

/// Save adversarial example as PNG image
pub fn save_adversarial_example<B: Backend>(
    tensor: &Tensor<B, 4>,
    path: &Path,
    idx: usize,
    true_label: usize,
    target_label: usize,
) -> anyhow::Result<()> {
    use image::{ImageBuffer, Luma, Rgb};

    // Create directory if it doesn't exist
    std::fs::create_dir_all(path)?;

    // Get tensor data
    let data = tensor.to_data();
    let values: Vec<f32> = data.iter::<f32>().collect();
    let [_batch, channels, height, width] = tensor.dims();

    // Create filename
    let filename = format!("adv_{}_true{}_target{}.png", idx, true_label, target_label);
    let filepath = path.join(filename);

    // Normalize values to [0, 255] range for visualization
    let min_val = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let range = max_val - min_val;

    let normalized: Vec<u8> = values
        .iter()
        .map(|&v| {
            if range > 1e-8 {
                ((v - min_val) / range * 255.0).clamp(0.0, 255.0) as u8
            } else {
                128u8
            }
        })
        .collect();

    // Save based on number of channels
    match channels {
        1 => {
            // Grayscale (MNIST)
            let mut img_buffer = ImageBuffer::<Luma<u8>, Vec<u8>>::new(width as u32, height as u32);
            for y in 0..height {
                for x in 0..width {
                    let idx = y * width + x;
                    img_buffer.put_pixel(x as u32, y as u32, Luma([normalized[idx]]));
                }
            }
            img_buffer.save(&filepath)?;
        }
        3 => {
            // RGB (CIFAR-10)
            let mut img_buffer = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(width as u32, height as u32);
            for y in 0..height {
                for x in 0..width {
                    let r = normalized[y * width + x];
                    let g = normalized[height * width + y * width + x];
                    let b = normalized[2 * height * width + y * width + x];
                    img_buffer.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
                }
            }
            img_buffer.save(&filepath)?;
        }
        _ => {
            return Err(anyhow::anyhow!(
                "Unsupported number of channels: {}",
                channels
            ));
        }
    }

    Ok(())
}

/// Generate a full text report
pub fn generate_report(
    model_name: &str,
    dataset_name: &str,
    num_samples: usize,
    confidence: f64,
    learning_rate: f64,
    max_iterations: usize,
    binary_search_steps: usize,
    initial_const: f64,
    metrics: &AttackMetrics,
) -> String {
    format!(
        r#"===============================================
C&W L2 Attack Results
===============================================
Model: {}
Dataset: {}
Samples: {}
Confidence (kappa): {}

--- Configuration ---
Learning Rate: {}
Max Iterations: {}
Binary Search Steps: {}
Initial Const: {}

{}
===============================================
"#,
        model_name,
        dataset_name,
        num_samples,
        confidence,
        learning_rate,
        max_iterations,
        binary_search_steps,
        initial_const,
        metrics.to_text_report()
    )
}

/// Save results to CSV file
pub fn save_results_to_csv(results: &[AttackResult], path: &Path) -> std::io::Result<()> {
    let mut file = File::create(path)?;

    // Write header
    writeln!(file, "{}", AttackResult::csv_header())?;

    // Write rows
    for result in results {
        writeln!(file, "{}", result.to_csv_row())?;
    }

    Ok(())
}

/// Save report to text file
pub fn save_report(report: &str, path: &Path) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    file.write_all(report.as_bytes())?;
    Ok(())
}
