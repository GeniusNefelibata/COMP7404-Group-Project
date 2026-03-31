use super::{AttackConfig, AttackResult};
use burn::{prelude::*, tensor::backend::AutodiffBackend};
use std::time::{Duration, Instant};

/// Adam optimizer state for a single tensor
#[derive(Clone)]
struct AdamState<B: burn::tensor::backend::Backend, const D: usize> {
    /// The number of iterations aggregated.
    time: usize,
    /// First moment (momentum)
    moment_1: Tensor<B, D>,
    /// Second moment (velocity)
    moment_2: Tensor<B, D>,
}

/// Simple Adam optimizer for tensor optimization
struct AdamOptimizer {
    /// Learning rate
    lr: f64,
    /// Beta 1 (first moment decay rate)
    beta_1: f64,
    /// Beta 2 (second moment decay rate)
    beta_2: f64,
    /// Epsilon for numerical stability
    epsilon: f64,
}

impl AdamOptimizer {
    /// Create a new Adam optimizer with default parameters
    fn new(lr: f64) -> Self {
        Self {
            lr,
            beta_1: 0.9,
            beta_2: 0.999,
            epsilon: 1e-8,
        }
    }

    /// Perform an Adam optimization step
    fn step<B: burn::tensor::backend::Backend, const D: usize>(
        &self,
        tensor: Tensor<B, D>,
        grad: Tensor<B, D>,
        state: Option<AdamState<B, D>>,
    ) -> (Tensor<B, D>, AdamState<B, D>) {
        let state = if let Some(mut state) = state {
            // Update biased first moment estimate
            let factor_1 = 1.0 - self.beta_1;
            state.moment_1 = state
                .moment_1
                .mul_scalar(self.beta_1)
                .add(grad.clone().mul_scalar(factor_1));

            // Update biased second raw moment estimate
            let factor_2 = 1.0 - self.beta_2;
            state.moment_2 = state
                .moment_2
                .mul_scalar(self.beta_2)
                .add(grad.square().mul_scalar(factor_2));

            state.time += 1;
            state
        } else {
            // Initialize state
            let factor_1 = 1.0 - self.beta_1;
            let moment_1 = grad.clone().mul_scalar(factor_1);

            let factor_2 = 1.0 - self.beta_2;
            let moment_2 = grad.square().mul_scalar(factor_2);

            AdamState {
                time: 1,
                moment_1,
                moment_2,
            }
        };

        // Bias correction
        let time = state.time as i32;
        let bias_correction_1 = 1.0 - self.beta_1.powi(time);
        let bias_correction_2 = 1.0 - self.beta_2.powi(time);

        // Compute step size
        let step_size = self.lr * (bias_correction_2.sqrt() / bias_correction_1);

        // Compute update: lr * m_t / (sqrt(v_t) + epsilon)
        let denom = state.moment_2.clone().sqrt().add_scalar(self.epsilon);
        let update = state.moment_1.clone().div(denom).mul_scalar(step_size);

        // Apply update
        let new_tensor = tensor.sub(update);

        (new_tensor, state)
    }
}

/// C&W L2 Attack implementation
pub struct CWL2Attack<B: AutodiffBackend> {
    config: AttackConfig,
    device: B::Device,
}

impl<B: AutodiffBackend> CWL2Attack<B> {
    pub fn new(config: AttackConfig, device: B::Device) -> Self {
        Self { config, device }
    }

    /// Tanh transform: maps w from unbounded space to [0, 1]
    /// x = 0.5 * (tanh(w) + 1)
    fn tanh_transform(&self, w: Tensor<B, 4>) -> Tensor<B, 4> {
        let tanh_w = w.tanh();
        tanh_w.mul_scalar(0.5).add_scalar(0.5)
    }

    /// Inverse tanh transform: maps x from [0, 1] to unbounded space
    /// w = arctanh(2 * x - 1)
    fn inverse_tanh_transform(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // Clamp to avoid numerical issues
        let x_clamped = x.clamp(self.config.clip_min + 1e-7, self.config.clip_max - 1e-7);
        let scaled = x_clamped.mul_scalar(2.0).sub_scalar(1.0);
        scaled.atanh()
    }

    /// Compute margin loss for targeted attack
    /// loss = max(max_other_logits - target_logits, -confidence)
    fn margin_loss(&self, logits: Tensor<B, 2>, target: usize) -> Tensor<B, 1> {
        let [batch_size, num_classes] = logits.dims();

        // Get target class logits - select returns a tensor with the selected indices
        let target_tensor =
            Tensor::<B, 1, Int>::from_data(TensorData::from([target as i64]), &self.device);
        let target_logits = logits.clone().select(1, target_tensor);

        // Compute max of other classes
        let mut max_other_vec = Vec::new();
        for c in 0..num_classes {
            if c != target {
                let c_tensor =
                    Tensor::<B, 1, Int>::from_data(TensorData::from([c as i64]), &self.device);
                let class_logits = logits.clone().select(1, c_tensor);
                max_other_vec.push(class_logits);
            }
        }

        // Stack and find max
        let mut max_other = max_other_vec[0].clone();
        for i in 1..max_other_vec.len() {
            max_other = max_other.max_pair(max_other_vec[i].clone());
        }

        // margin_loss = max(max_other - target_logits, -confidence)
        let diff = max_other.sub(target_logits);
        let neg_confidence = -self.config.confidence;
        let confidence_tensor =
            Tensor::<B, 1>::zeros([batch_size], &self.device).add_scalar(neg_confidence);

        // Need to reshape confidence to match diff
        let confidence_reshaped = confidence_tensor.reshape([batch_size, 1]);
        let diff_reshaped = diff.reshape([batch_size, 1]);
        let result = diff_reshaped.max_pair(confidence_reshaped);
        result.reshape([batch_size])
    }

    /// Attack a single image
    pub fn attack_single<F>(
        &self,
        image: Tensor<B, 4>,
        true_label: usize,
        target_label: usize,
        idx: usize,
        model_forward: F,
    ) -> (AttackResult, Option<Tensor<B, 4>>)
    where
        F: Fn(Tensor<B, 4>) -> Tensor<B, 2>,
    {
        let start_time = Instant::now();
        let timeout = Duration::from_secs(self.config.timeout_secs);

        let mut best_l2 = f64::INFINITY;
        let mut best_adv: Option<Tensor<B, 4>> = None;
        let mut found_adv = false;
        let mut best_pred: Option<usize> = None;

        let mut c = self.config.c.unwrap_or(self.config.initial_const);
        let mut lower_bound: f64 = 0.0;
        let mut upper_bound: f64 = 1e10;

        let search_steps = if self.config.c.is_some() {
            1
        } else {
            self.config.binary_search_steps
        };

        for _ in 0..search_steps {
            // Initialize w from image using inverse tanh transform
            let w_init = self.inverse_tanh_transform(image.clone());
            let mut w_data = w_init.into_data();

            // Create Adam optimizer
            let adam = AdamOptimizer::new(self.config.learning_rate);
            let mut adam_state: Option<AdamState<B, 4>> = None;

            for iteration in 0..self.config.max_iterations {
                // Check timeout
                if start_time.elapsed() > timeout {
                    println!(); // newline after progress
                    return self.build_result(
                        idx,
                        true_label,
                        target_label,
                        None,
                        f64::INFINITY,
                        c,
                        start_time.elapsed(),
                        best_adv,
                    );
                }

                // Progress indicator every 100 iterations
                if iteration % 100 == 0 {
                    print!(
                        "\r    Progress: {}/{} iterations",
                        iteration, self.config.max_iterations
                    );
                    std::io::Write::flush(&mut std::io::stdout()).unwrap();
                }

                // Create tensor from w_data with gradient tracking
                let w = Tensor::<B, 4>::from_data(w_data.clone(), &self.device).require_grad();

                // Transform w to adversarial image
                let x_adv = self.tanh_transform(w.clone());

                // Forward pass
                let logits = model_forward(x_adv.clone());

                // Compute losses
                let margin_loss = self.margin_loss(logits.clone(), target_label);
                let l2_dist = self.compute_l2_distance(&x_adv, &image);

                // Total loss: L2_distance + c * margin_loss
                let c_tensor = Tensor::<B, 1>::ones([1], &self.device).mul_scalar(c);
                let total_loss = l2_dist.clone().add(margin_loss.clone().mul(c_tensor));

                // Store values for debug before backward
                let _total_val = total_loss.clone().into_scalar().to_f64();
                let margin_val = margin_loss.clone().into_scalar().to_f64();
                let l2_val = l2_dist.clone().into_scalar().to_f64();
                let pred = logits.argmax(1).into_scalar().to_usize();

                // Backward pass to get gradients
                let grads = total_loss.backward();

                // Get gradient for w
                let w_grad = w.grad(&grads).expect("Gradient not found");
                let w_grad_tensor: Tensor<B, 4> =
                    Tensor::from_data(w_grad.into_data(), &self.device);

                // Convert w to tensor for Adam step
                let w_tensor = Tensor::<B, 4>::from_data(w_data.clone(), &self.device);

                // Adam optimizer step
                let (w_new, new_state) = adam.step(w_tensor, w_grad_tensor, adam_state);
                w_data = w_new.into_data();
                adam_state = Some(new_state);

                if margin_val <= 0.0 && l2_val < best_l2 {
                    best_l2 = l2_val;
                    best_adv = Some(x_adv.clone());
                    best_pred = Some(pred);
                    found_adv = true;

                    if self.config.abort_early {
                        break;
                    }
                }
            }

            // Binary search for c
            if self.config.c.is_none() {
                if found_adv {
                    upper_bound = if upper_bound < c { upper_bound } else { c };
                    c = (lower_bound + upper_bound) / 2.0;
                } else {
                    lower_bound = if lower_bound > c { lower_bound } else { c };
                    if upper_bound < 1e10 {
                        c = (lower_bound + upper_bound) / 2.0;
                    } else {
                        c *= 10.0;
                    }
                }
            }
        }

        self.build_result(
            idx,
            true_label,
            target_label,
            best_pred,
            if best_l2 == f64::INFINITY {
                best_l2
            } else {
                best_l2.sqrt()
            },
            c,
            start_time.elapsed(),
            best_adv,
        )
    }

    /// Compute L2 distance between two images
    fn compute_l2_distance(&self, x1: &Tensor<B, 4>, x2: &Tensor<B, 4>) -> Tensor<B, 1> {
        let diff = x1.clone().sub(x2.clone());
        let diff_clone = diff.clone();
        let squared = diff.mul(diff_clone);
        // Sum over all dimensions except batch
        let [_b, c, h, w] = squared.dims();
        squared.sum().div_scalar((c * h * w) as f64)
    }

    /// Build attack result
    fn build_result(
        &self,
        idx: usize,
        true_label: usize,
        target_label: usize,
        pred_label: Option<usize>,
        l2_distance: f64,
        best_c: f64,
        elapsed: Duration,
        best_adv: Option<Tensor<B, 4>>,
    ) -> (AttackResult, Option<Tensor<B, 4>>) {
        let success = pred_label.map_or(false, |p| p == target_label);

        let result = AttackResult {
            image_idx: idx,
            true_label,
            target_label,
            pred_label,
            success,
            l2_distance,
            best_c,
            time_seconds: elapsed.as_secs_f64(),
        };

        (result, best_adv)
    }
}
