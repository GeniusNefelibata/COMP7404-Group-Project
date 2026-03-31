use clap::{Parser, ValueEnum};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum Backend {
    NdArray,
    Wgpu,
}

#[derive(Parser, Debug)]
#[command(name = "cw-l2-attack")]
#[command(about = "C&W L2 Adversarial Attack using Burn framework")]
pub struct Cli {
    /// Dataset to attack (mnist or cifar10)
    #[arg(short, long, value_enum)]
    pub dataset: Option<Dataset>,

    /// Dataset to attack (mnist or cifar10)
    #[arg(short, long)]
    pub backend: Option<Backend>,

    /// Path to model checkpoint
    #[arg(short, long)]
    pub model_path: PathBuf,

    /// Path to single image file (alternative to dataset)
    #[arg(short, long)]
    pub image: Option<PathBuf>,

    /// Target class for single image attack
    #[arg(short, long)]
    pub target_class: Option<usize>,

    /// Number of samples to attack (sequential selection from start)
    #[arg(short, long, default_value = "1000")]
    pub num_samples: usize,

    /// Output directory for results
    #[arg(short, long, default_value = "./results")]
    pub output: PathBuf,

    /// Fixed c value (if not specified, uses binary search)
    #[arg(long)]
    pub c: Option<f64>,

    /// Confidence parameter (kappa)
    #[arg(long, default_value = "0.0")]
    pub confidence: f64,

    /// Learning rate for Adam optimizer
    #[arg(long, default_value = "0.01")]
    pub learning_rate: f64,

    /// Maximum iterations per binary search step
    #[arg(long, default_value = "1000")]
    pub max_iterations: usize,

    /// Number of binary search steps
    #[arg(long, default_value = "9")]
    pub binary_search_steps: usize,

    /// Initial const value for binary search
    #[arg(long, default_value = "0.001")]
    pub initial_const: f64,

    /// Whether to perform targeted attack
    #[arg(long, default_value = "true")]
    pub targeted: bool,

    /// Data directory for datasets
    #[arg(long, default_value = "./data")]
    pub data_dir: PathBuf,

    /// Directory to save adversarial examples (only saved on successful attacks)
    #[arg(long)]
    pub save_adv: Option<PathBuf>,

    /// Timeout in seconds per sample (default: 300)
    #[arg(long, default_value = "300")]
    pub timeout: u64,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq)]
pub enum Dataset {
    Mnist,
    Cifar10,
    ImageNet,
}

impl Cli {
    /// Validate arguments
    pub fn validate(&self) -> anyhow::Result<()> {
        // Either dataset or single image must be specified
        if self.dataset.is_none() && self.image.is_none() {
            anyhow::bail!("Either --dataset or --image must be specified");
        }

        // If single image is specified without dataset, we need dataset to determine model type
        if self.image.is_some() && self.dataset.is_none() {
            anyhow::bail!("--dataset must be specified with --image to determine model type");
        }

        // If single image is specified, target class is required
        if self.image.is_some() && self.target_class.is_none() {
            anyhow::bail!("--target-class is required when using --image");
        }

        // Model file must exist
        if !self.model_path.exists() {
            anyhow::bail!("Model file not found: {:?}", self.model_path);
        }

        Ok(())
    }
}
