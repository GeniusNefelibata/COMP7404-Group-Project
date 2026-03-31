use burn::module::Module;
use burn_store::{ModuleSnapshot, PytorchStore, SafetensorsStore};
use std::path::Path;

/// Load model weights from a checkpoint file
///
/// Supports both PyTorch .pth files and Safetensors .safetensors files
/// The file format is detected automatically based on the extension.
/// For PyTorch files, it auto-detects whether weights are wrapped in 'model_state_dict' or stored directly.
pub fn load_model_from_checkpoint<B, M>(model: &mut M, checkpoint_path: &str) -> anyhow::Result<()>
where
    M: Module<B>,
    B: burn::tensor::backend::Backend,
{
    let path = Path::new(checkpoint_path);
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match ext {
        "safetensors" => {
            // Load from Safetensors format
            let mut store = SafetensorsStore::from_file(checkpoint_path).allow_partial(true);
            model.load_from(&mut store)?;
        }
        "pth" | "pt" | _ => {
            // Load from PyTorch format
            // Check if this looks like an Inception V3 model (direct weight storage)
            let filename = path
                .file_name()
                .and_then(|f| f.to_str())
                .unwrap_or("")
                .to_lowercase();

            if filename.contains("inception") || filename.contains("google") {
                // Inception V3 models have weights stored directly, not in model_state_dict
                let mut store = PytorchStore::from_file(checkpoint_path).allow_partial(true);
                model.load_from(&mut store)?;
            } else {
                // Other models use model_state_dict wrapper
                let mut store = PytorchStore::from_file(checkpoint_path)
                    .with_top_level_key("model_state_dict")
                    .allow_partial(true);
                model.load_from(&mut store)?;
            }
        }
    }

    Ok(())
}
