mod attack;
mod checkpoint;
mod cli;
mod data;
mod models;

use crate::{
    attack::{
        AttackConfig, AttackResult,
        cw_l2::CWL2Attack,
        utils::{self, AttackMetrics},
    },
    cli::{Cli, Dataset},
    data::{
        AttackDataset, cifar::CifarAttackDataset, imagenet::ImageNetAttackDataset,
        mnist::MnistAttackDataset,
    },
    models::{cifar::CifarModel, inception_v3::InceptionV3Config, mnist::MnistModel},
};
use clap::Parser;
use std::fs;
use std::path::Path;

/// Detect model type from checkpoint filename
/// Returns the detected dataset type based on filename keywords
fn detect_model_type_from_checkpoint(checkpoint_path: &Path) -> Option<Dataset> {
    let filename = checkpoint_path
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or("")
        .to_lowercase();

    if filename.contains("cifar") {
        Some(Dataset::Cifar10)
    } else if filename.contains("mnist") {
        Some(Dataset::Mnist)
    } else if filename.contains("inception") || filename.contains("imagenet") {
        Some(Dataset::ImageNet)
    } else {
        None
    }
}

fn main() -> anyhow::Result<()> {
    // Parse command line arguments
    let cli = Cli::parse();
    cli.validate()?;
    match cli.backend.unwrap_or(cli::Backend::Wgpu) {
        cli::Backend::Wgpu => man::<burn::backend::Autodiff<burn::backend::Wgpu>>(cli),
        cli::Backend::NdArray => man::<burn::backend::Autodiff<burn::backend::NdArray>>(cli),
    }
}

fn man<AutodiffBackend>(cli: Cli) -> anyhow::Result<()>
where
    AutodiffBackend: burn::tensor::backend::AutodiffBackend,
{
    // Create output directory
    fs::create_dir_all(&cli.output)?;

    let device = AutodiffBackend::Device::default();

    println!("C&W L2 Attack");
    println!("==============");
    println!("Model: {:?}", cli.model_path);
    println!("Device: {:?}", device);

    // Auto-detect model type from checkpoint filename
    let detected_model_type = detect_model_type_from_checkpoint(&cli.model_path);
    if let Some(model_type) = &detected_model_type {
        println!("Detected model type from checkpoint: {:?}", model_type);
    }

    // Run attack based on mode
    match (&cli.dataset, &cli.image) {
        (Some(dataset), None) => {
            // Use detected model type if available, otherwise fall back to dataset
            let model_type = detected_model_type.unwrap_or(*dataset);
            if model_type != *dataset {
                println!(
                    "Warning: Dataset is {:?} but model checkpoint suggests {:?}. Using {:?} model.",
                    dataset, model_type, model_type
                );
            }
            attack_dataset::<AutodiffBackend>(&cli, *dataset, model_type, &device)?;
        }
        (Some(dataset), Some(image_path)) => {
            // Use detected model type if available, otherwise fall back to dataset
            let model_type = detected_model_type.unwrap_or(*dataset);
            if model_type != *dataset {
                println!(
                    "Warning: Dataset is {:?} but model checkpoint suggests {:?}. Using {:?} model.",
                    dataset, model_type, model_type
                );
            }
            attack_single_image::<AutodiffBackend>(
                &cli, *dataset, model_type, image_path, &device,
            )?;
        }
        (None, Some(_)) => {
            // This case is handled by validation which requires dataset with image
            unreachable!()
        }
        (None, None) => {
            anyhow::bail!("Either --dataset or --image must be specified");
        }
    }

    Ok(())
}

fn attack_dataset<AutodiffBackend>(
    cli: &Cli,
    dataset: Dataset,
    model_type: Dataset,
    device: &AutodiffBackend::Device,
) -> anyhow::Result<()>
where
    AutodiffBackend: burn::tensor::backend::AutodiffBackend,
{
    println!("\nAttacking dataset: {:?}", dataset);
    println!("Using model type: {:?}", model_type);
    println!("Number of samples: {}", cli.num_samples);

    // Create attack configuration
    let attack_config = AttackConfig {
        confidence: cli.confidence,
        learning_rate: cli.learning_rate,
        max_iterations: cli.max_iterations,
        binary_search_steps: cli.binary_search_steps,
        initial_const: cli.initial_const,
        c: cli.c,
        abort_early: true,
        clip_min: 0.0,
        clip_max: 1.0,
        timeout_secs: cli.timeout,
    };

    match model_type {
        Dataset::Mnist => {
            // Load MNIST dataset (use model_type to determine which dataset to load)
            let dataset = MnistAttackDataset::<AutodiffBackend>::test(
                cli.data_dir.to_str().unwrap(),
                device.clone(),
            )?;

            // Load model
            let mut model: MnistModel<AutodiffBackend> = MnistModel::new(device);
            // Load model weights from cli.model_path
            println!("Loading model weights from: {:?}", cli.model_path);
            checkpoint::load_model_from_checkpoint(&mut model, cli.model_path.to_str().unwrap())?;

            // Run attack
            let attack = CWL2Attack::new(attack_config, device.clone());
            let mut results = Vec::new();

            let num_samples = cli.num_samples.min(dataset.len());

            for idx in 0..num_samples {
                let sample = dataset.get(idx).unwrap();

                // Select target class (different from true label)
                let target_label = if let Some(class) = cli.target_class {
                    class
                } else {
                    (sample.label + 1) % 10
                };

                print!("[{}/{}] Sample {}: ", idx + 1, num_samples, idx);

                let (result, adv_tensor) =
                    attack.attack_single(sample.image, sample.label, target_label, idx, |input| {
                        model.forward(input)
                    });

                println!(
                    "true={}, target={}, success={}, l2={:.4}, time={:.2}s",
                    result.true_label,
                    result.target_label,
                    result.success,
                    result.l2_distance,
                    result.time_seconds
                );

                // Save adversarial example if attack was successful and --save-adv is specified
                if result.success {
                    if let (Some(save_dir), Some(adv)) = (&cli.save_adv, adv_tensor) {
                        if let Err(e) = utils::save_adversarial_example(
                            &adv,
                            save_dir,
                            idx,
                            sample.label,
                            target_label,
                        ) {
                            eprintln!("  Warning: Failed to save adversarial example: {}", e);
                        } else {
                            println!("  Saved adversarial example to {:?}", save_dir);
                        }
                    }
                }

                results.push(result);
            }

            // Compute metrics and save results
            save_results(&results, cli, "mnist")?;
        }
        Dataset::Cifar10 => {
            // Load CIFAR-10 dataset
            let dataset = CifarAttackDataset::<AutodiffBackend>::test(
                cli.data_dir.to_str().unwrap(),
                device.clone(),
            )?;

            // Load model
            let mut model: CifarModel<AutodiffBackend> = CifarModel::new(device);
            // Load model weights from cli.model_path
            println!("Loading model weights from: {:?}", cli.model_path);
            checkpoint::load_model_from_checkpoint(&mut model, cli.model_path.to_str().unwrap())?;

            // Run attack
            let attack = CWL2Attack::new(attack_config, device.clone());
            let mut results = Vec::new();

            let num_samples = cli.num_samples.min(dataset.len());

            for idx in 0..num_samples {
                let sample = dataset.get(idx).unwrap();

                // Select target class (different from true label)
                let target_label = if let Some(class) = cli.target_class {
                    class
                } else {
                    (sample.label + 1) % 10
                };

                print!("[{}/{}] Sample {}: ", idx + 1, num_samples, idx);

                let (result, adv_tensor) =
                    attack.attack_single(sample.image, sample.label, target_label, idx, |input| {
                        model.forward(input)
                    });

                println!(
                    "true={}, target={}, success={}, time={:.2}s, l2={:.}",
                    result.true_label,
                    result.target_label,
                    result.success,
                    result.time_seconds,
                    result.l2_distance,
                );

                // Save adversarial example if attack was successful and --save-adv is specified
                if result.success {
                    if let (Some(save_dir), Some(adv)) = (&cli.save_adv, adv_tensor) {
                        if let Err(e) = utils::save_adversarial_example(
                            &adv,
                            save_dir,
                            idx,
                            sample.label,
                            target_label,
                        ) {
                            eprintln!("  Warning: Failed to save adversarial example: {}", e);
                        } else {
                            println!("  Saved adversarial example to {:?}", save_dir);
                        }
                    }
                }

                results.push(result);
            }

            // Compute metrics and save results
            save_results(&results, cli, "cifar10")?;
        }
        Dataset::ImageNet => {
            // Load ImageNet dataset
            let dataset = ImageNetAttackDataset::<AutodiffBackend>::test(
                cli.data_dir.to_str().unwrap(),
                device.clone(),
            )?;

            // Load model
            let config = InceptionV3Config::default();
            let mut model = config.init::<AutodiffBackend>(device);
            // Load model weights from cli.model_path
            println!("Loading model weights from: {:?}", cli.model_path);
            checkpoint::load_model_from_checkpoint(&mut model, cli.model_path.to_str().unwrap())?;

            // Run attack
            let attack = CWL2Attack::new(attack_config, device.clone());
            let mut results = Vec::new();

            let num_samples = cli.num_samples.min(dataset.len());

            for idx in 0..num_samples {
                let sample = dataset.get(idx).unwrap();

                // Select target class (different from true label)
                let target_label = if let Some(class) = cli.target_class {
                    class
                } else {
                    (sample.label + 1) % 1000
                };

                print!("[{}/{}] Sample {}: ", idx + 1, num_samples, idx);

                let (result, adv_tensor) =
                    attack.attack_single(sample.image, sample.label, target_label, idx, |input| {
                        model.forward(input, false).0 // Returns (logits, aux_logits)
                    });

                println!(
                    "true={}, target={}, success={}, time={:.2}s, l2={:.}",
                    result.true_label,
                    result.target_label,
                    result.success,
                    result.time_seconds,
                    result.l2_distance,
                );

                // Save adversarial example if attack was successful and --save-adv is specified
                if result.success {
                    if let (Some(save_dir), Some(adv)) = (&cli.save_adv, adv_tensor) {
                        if let Err(e) = utils::save_adversarial_example(
                            &adv,
                            save_dir,
                            idx,
                            sample.label,
                            target_label,
                        ) {
                            eprintln!("  Warning: Failed to save adversarial example: {}", e);
                        } else {
                            println!("  Saved adversarial example to {:?}", save_dir);
                        }
                    }
                }

                results.push(result);
            }

            // Compute metrics and save results
            save_results(&results, cli, "imagenet")?;
        }
    }

    Ok(())
}

fn attack_single_image<AutodiffBackend: burn::tensor::backend::AutodiffBackend>(
    cli: &Cli,
    dataset: Dataset,
    model_type: Dataset,
    image_path: &std::path::Path,
    device: &AutodiffBackend::Device,
) -> anyhow::Result<()> {
    println!("\nAttacking single image: {:?}", image_path);
    println!("Using model type: {:?}", model_type);

    let target_label = cli.target_class.unwrap();
    println!("Target class: {}", target_label);

    // Load image based on dataset type
    let image = match dataset {
        Dataset::Mnist => {
            data::mnist::load_mnist_image::<AutodiffBackend>(image_path.to_str().unwrap(), device)?
        }
        Dataset::Cifar10 => {
            data::cifar::load_cifar_image::<AutodiffBackend>(image_path.to_str().unwrap(), device)?
        }
        Dataset::ImageNet => data::imagenet::load_imagenet_image::<AutodiffBackend>(
            image_path.to_str().unwrap(),
            device,
        )?,
    };

    // Create attack configuration
    let attack_config = AttackConfig {
        confidence: cli.confidence,
        learning_rate: cli.learning_rate,
        max_iterations: cli.max_iterations,
        binary_search_steps: cli.binary_search_steps,
        initial_const: cli.initial_const,
        c: cli.c,
        abort_early: true,
        clip_min: 0.0,
        clip_max: 1.0,
        timeout_secs: cli.timeout,
    };

    // Load appropriate model and run attack
    let (result, _) = match model_type {
        Dataset::Mnist => {
            let mut model: MnistModel<AutodiffBackend> = MnistModel::new(device);
            // Load model weights
            println!("Loading model weights from: {:?}", cli.model_path);
            checkpoint::load_model_from_checkpoint(&mut model, cli.model_path.to_str().unwrap())?;

            let attack = CWL2Attack::new(attack_config, device.clone());
            attack.attack_single(
                image,
                0, // Unknown true label for single image
                target_label,
                0,
                |input| model.forward(input),
            )
        }
        Dataset::Cifar10 => {
            let mut model: CifarModel<AutodiffBackend> = CifarModel::new(device);
            // Load model weights
            println!("Loading model weights from: {:?}", cli.model_path);
            checkpoint::load_model_from_checkpoint(&mut model, cli.model_path.to_str().unwrap())?;

            let attack = CWL2Attack::new(attack_config, device.clone());
            attack.attack_single(
                image,
                0, // Unknown true label for single image
                target_label,
                0,
                |input| model.forward(input),
            )
        }
        Dataset::ImageNet => {
            let config = InceptionV3Config::default();
            let mut model = config.init::<AutodiffBackend>(device);
            // Load model weights
            println!("Loading model weights from: {:?}", cli.model_path);
            checkpoint::load_model_from_checkpoint(&mut model, cli.model_path.to_str().unwrap())?;

            let attack = CWL2Attack::new(attack_config, device.clone());
            attack.attack_single(
                image,
                0, // Unknown true label for single image
                target_label,
                0,
                |input| model.forward(input, false).0, // Returns (logits, aux_logits)
            )
        }
    };

    println!("\nAttack Result:");
    println!("  Target: {}", result.target_label);
    println!("  Predicted: {:?}", result.pred_label);
    println!("  Success: {}", result.success);
    println!("  L2 Distance: {:.4}", result.l2_distance);
    println!("  Best c: {:.6}", result.best_c);
    println!("  Time: {:.2}s", result.time_seconds);

    Ok(())
}

fn save_results(results: &[AttackResult], cli: &Cli, dataset_name: &str) -> anyhow::Result<()> {
    // Compute metrics
    let metrics = AttackMetrics::from_results(results);

    println!("\n{}", metrics.to_text_report());

    // Save CSV
    let csv_path = cli
        .output
        .join(format!("cw_l2_{}_results.csv", dataset_name));
    utils::save_results_to_csv(results, &csv_path)?;
    println!("Results saved to: {:?}", csv_path);

    // Save text report
    let report = utils::generate_report(
        &cli.model_path.file_name().unwrap().to_string_lossy(),
        dataset_name,
        results.len(),
        cli.confidence,
        cli.learning_rate,
        cli.max_iterations,
        cli.binary_search_steps,
        cli.initial_const,
        &metrics,
    );

    let report_path = cli
        .output
        .join(format!("cw_l2_{}_report.txt", dataset_name));
    utils::save_report(&report, &report_path)?;
    println!("Report saved to: {:?}", report_path);

    Ok(())
}
