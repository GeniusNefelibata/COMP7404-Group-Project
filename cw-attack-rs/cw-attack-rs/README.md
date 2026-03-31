# C&W L2 Attack Implementation in Burn

This project implements the Carlini & Wagner (C&W) L2 adversarial attack using the [Burn](https://github.com/tracel-ai/burn) deep learning framework in Rust.

## Features

- **C&W L2 Attack**: Full implementation of the Carlini & Wagner L2 attack with tanh transform and margin loss
- **Binary Search for c**: Automatic determination of optimal c value using binary search (when not specified)
- **Model Support**: Works with both MNIST and CIFAR-10 models
- **PyTorch Checkpoint Loading**: Direct loading of PyTorch `.pth` checkpoint files using `burn-store`'s `PytorchStore`
- **Flexible Input**: Supports both dataset testing and single image attacks
- **Sequential Selection**: Selects samples sequentially from the start of datasets (not random)
- **Output Formats**: CSV results and text reports (no JSON)

## Project Structure

```
src/
├── attack/
│   ├── cw_l2.rs      # Core C&W L2 attack implementation
│   ├── utils.rs      # Metrics computation and output formatting
│   └── mod.rs
├── checkpoint/
│   └── mod.rs        # PyTorch checkpoint loading using burn-store
├── cli.rs            # Command-line argument parsing
├── data/
│   ├── mnist.rs      # MNIST dataset loading
│   ├── cifar.rs      # CIFAR-10 dataset loading
│   └── mod.rs
├── models/
│   ├── mnist.rs      # MNIST ConvNet model
│   ├── cifar.rs      # CIFAR-10 ConvNet model
│   └── mod.rs
├── lib.rs
└── main.rs
```

## Building

```bash
cargo build --release
```

## Usage

### Attack Dataset

Attack 100 samples from MNIST dataset:

```bash
cw-l2-attack \
    --dataset mnist \
    --model-path reference/python-attack/ckpt/undistilled_mnist.pth \
    --num-samples 100 \
    --output ./results
```

Attack CIFAR-10 with custom parameters:

```bash
cw-l2-attack \
    --dataset cifar10 \
    --model-path reference/python-attack/ckpt/undistilled_cifar10.pth \
    --num-samples 100 \
    --confidence 0.0 \
    --learning-rate 0.01 \
    --max-iterations 1000 \
    --output ./results
```

### Attack Single Image

```bash
cw-l2-attack \
    --dataset mnist \
    --model-path reference/python-attack/ckpt/undistilled_mnist.pth \
    --image path/to/image.png \
    --target-class 3 \
    --output ./results
```

### Specify Fixed c Value

By default, the attack uses binary search to find the optimal c value. To use a fixed c:

```bash
cw-l2-attack \
    --dataset mnist \
    --model-path reference/python-attack/ckpt/undistilled_mnist.pth \
    --c 0.1 \
    --num-samples 100
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-d, --dataset` | Dataset to attack (mnist or cifar10) | - |
| `-m, --model-path` | Path to model checkpoint (.pth file) | **required** |
| `-i, --image` | Path to single image file | - |
| `-t, --target-class` | Target class for single image attack | - |
| `-n, --num-samples` | Number of samples to attack (sequential) | 1000 |
| `-o, --output` | Output directory for results | ./results |
| `--c` | Fixed c value (uses binary search if not set) | - |
| `--confidence` | Confidence parameter (kappa) | 0.0 |
| `--learning-rate` | Learning rate for Adam optimizer | 0.01 |
| `--max-iterations` | Maximum iterations per binary search step | 1000 |
| `--binary-search-steps` | Number of binary search steps | 9 |
| `--initial-const` | Initial const value for binary search | 0.001 |
| `--targeted` | Whether to perform targeted attack | true |
| `--data-dir` | Data directory for datasets | ./data |

## Implementation Details

### C&W L2 Attack Algorithm

The attack uses the following key components:

1. **Tanh Transform**: Maps the adversarial example to an unbounded space
   ```
   w = arctanh(2 * x - 1)
   x_adv = 0.5 * (tanh(w) + 1)
   ```

2. **Margin Loss**: Encourages the model to misclassify
   ```
   L_margin = max(Z(x_adv)_true - max(Z(x_adv)_i≠true) + κ, 0)
   ```

3. **L2 Distance**: Measures perturbation size
   ```
   L2 = ||x_adv - x||_2
   ```

4. **Combined Loss**:
   ```
   L = L2 + c * L_margin
   ```

### Binary Search for c

When c is not specified, the attack performs binary search:
- Starts with `c_lower = 0` and `c_upper = 1e10`
- For each binary search step:
  - Sets `c = (c_lower + c_upper) / 2`
  - Runs the attack with fixed c
  - If successful: `c_upper = c`, else `c_lower = c`
- Returns the best adversarial example found

### Model Architectures

**MNIST ConvNet**:
- Conv2d(1, 32, 3) + ReLU + MaxPool2d(2)
- Conv2d(32, 64, 3) + ReLU + MaxPool2d(2)
- Dropout(0.25)
- Linear(9216, 128) + ReLU
- Dropout(0.5)
- Linear(128, 10)

**CIFAR-10 ConvNet**:
- Conv2d(3, 64, 3) + ReLU
- Conv2d(64, 64, 3) + ReLU + MaxPool2d(2)
- Conv2d(64, 128, 3) + ReLU
- Conv2d(128, 128, 3) + ReLU + MaxPool2d(2)
- Flatten
- Linear(3200, 256) + ReLU
- Linear(256, 256) + ReLU
- Linear(256, 10)

## Dependencies

Key dependencies (without default features to avoid TUI):

```toml
burn = { version = "0.20.1", default-features = false, features = ["ndarray", "autodiff", "std", "train"] }
burn-ndarray = { version = "0.20.1", default-features = false, features = ["std"] }
burn-store = { version = "0.20.1", default-features = false, features = ["pytorch", "std"] }
```

## Output Files

The attack generates:
- `cw_l2_{dataset}_results.csv`: CSV file with attack results for each sample
- `cw_l2_{dataset}_report.txt`: Text report with summary statistics

## References

- [Carlini & Wagner (2017)](https://arxiv.org/abs/1608.04644): "Towards Evaluating the Robustness of Neural Networks"
- [Burn Framework](https://burn.dev/): Deep Learning Framework in Rust
