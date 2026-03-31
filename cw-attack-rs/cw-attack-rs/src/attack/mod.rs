pub mod cw_l2;
pub mod utils;

/// Result of a single attack
#[derive(Clone, Debug)]
pub struct AttackResult {
    pub image_idx: usize,
    pub true_label: usize,
    pub target_label: usize,
    pub pred_label: Option<usize>,
    pub success: bool,
    pub l2_distance: f64,
    pub best_c: f64,
    pub time_seconds: f64,
}

impl AttackResult {
    /// Convert to CSV row
    pub fn to_csv_row(&self) -> String {
        format!(
            "{},{},{},{},{},{:.},{:.},{:.3}",
            self.image_idx,
            self.true_label,
            self.target_label,
            self.pred_label.map_or("".to_string(), |v| v.to_string()),
            self.success,
            self.l2_distance,
            self.best_c,
            self.time_seconds
        )
    }

    /// CSV header
    pub fn csv_header() -> &'static str {
        "image_idx,true_label,target_label,pred_label,success,l2_distance,best_c,time_seconds"
    }
}

/// Configuration for C&W L2 attack
#[derive(Clone, Debug)]
pub struct AttackConfig {
    pub confidence: f64,
    pub learning_rate: f64,
    pub max_iterations: usize,
    pub binary_search_steps: usize,
    pub initial_const: f64,
    pub c: Option<f64>, // If None, use binary search
    pub abort_early: bool,
    pub clip_min: f64,
    pub clip_max: f64,
    pub timeout_secs: u64,
}

impl Default for AttackConfig {
    fn default() -> Self {
        Self {
            confidence: 0.0,
            learning_rate: 0.01,
            max_iterations: 1000,
            binary_search_steps: 9,
            initial_const: 0.001,
            c: None,
            abort_early: true,
            clip_min: 0.0,
            clip_max: 1.0,
            timeout_secs: 300,
        }
    }
}
