import os
from dataclasses import dataclass, field


def _env_int(name, default):
    return int(os.getenv(name, str(default)))


def _env_float(name, default):
    return float(os.getenv(name, str(default)))


@dataclass
class CFG:
    seed: int = 7

    # Spatial resolution. GRID can be 6, 8, 10, ... as long as GRID^2 >= 33.
    grid: int = field(default_factory=lambda: _env_int("GRID", 6))
    tile: int = field(default_factory=lambda: _env_int("TILE", 32))
    preview_cell: int = 3

    # Fixed downstream graph used by the controlled proof-of-concept.
    source_node: int = 0
    goal_node: int = 13
    n_edges: int = 33
    n_paths: int = 81

    risk_normal: float = 0.0
    risk_rough: float = 0.35
    risk_hazard: float = 1.00
    blocked_penalty: float = 20.0
    preview_abnormal_penalty: float = 0.70

    # Scene generation.
    critical_scene_fraction: float = 0.70
    critical_min_coarse_regret: float = 0.04
    max_generation_attempts: int = 200
    edge_abnormal_prob: float = 0.70
    rough_given_abnormal: float = 0.55
    hazard_given_abnormal: float = 0.37
    blocked_given_abnormal: float = 0.08

    n_train: int = 5000
    n_val: int = 1000
    n_test: int = 1000

    # Vision.
    feat_dim: int = 64
    high_patch_size: int = 4
    high_tokens_per_crop: int = 64

    # Shared Transformer flow field.
    hidden_dim: int = 128
    transformer_layers: int = 2
    transformer_heads: int = 4

    # Differentiable teacher decision surrogate.
    teacher_tau: float = 0.15
    grad_eps: float = 1e-8

    # Fixed visual acquisition budget.
    budget_frac: float = 0.15

    # Fixed-step neurodynamic integration of the learned flow field.
    nd_steps: int = 4
    nd_beta_max: float = 0.80

    # Stage 0.
    batch_size: int = 32
    lr_perception: float = 3e-4
    wd: float = 1e-4
    epochs_perception: int = 20

    # Stage I: gradient-distilled flow matching.
    lr_flow: float = 1e-4
    lr_preview: float = 1e-4
    epochs_flow: int = 40
    lambda_fm: float = 1.0
    lambda_gd: float = 1.0
    lambda_task: float = 0.25
    gd_magnitude_weight: float = 0.25

    # Stage II: outcome-only policy gradient. No DPO, no STOP, no critic.
    lr_rl: float = 3e-5
    epochs_rl: int = 20
    policy_temperature: float = 0.40
    entropy_coef: float = 0.001

    # Evaluation.
    # Decision evaluation: any path whose TRUE task cost is equal to the
    # full-information optimum within tolerance is counted as optimal, even if
    # its path ID differs. Regret remains the primary metric.
    optimal_cost_atol: float = 1e-6
    optimal_cost_rtol: float = 1e-6
    path_tol: float = 1e-6  # backward-compatible alias for old scripts
    selection_tie_tol: float = 1e-6

    # Gradient Student capacity-sweep defaults.
    student_head_temperature: float = 1.0
    gd_lambda_cos: float = 1.0
    gd_lambda_kl: float = 0.25
    gd_lambda_rank: float = 0.10
    gd_rank_margin: float = 0.02

    @property
    def n_patches(self):
        return self.grid * self.grid

    @property
    def image_size(self):
        return self.grid * self.tile

    @property
    def preview_size(self):
        return self.grid * self.preview_cell

    @property
    def high_crop_size(self):
        return self.tile

    @property
    def full_high_tokens(self):
        return self.n_patches * self.high_tokens_per_crop

    def visual_budget_k(self, frac=None):
        frac = self.budget_frac if frac is None else frac
        return max(1, min(self.n_patches, int(round(frac * self.n_patches))))


CFG = CFG()
if CFG.n_patches < CFG.n_edges:
    raise ValueError(f"GRID={CFG.grid} gives {CFG.n_patches} patches, but at least {CFG.n_edges} are required.")
