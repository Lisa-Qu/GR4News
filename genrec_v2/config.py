"""GenRec-V2 configuration."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GenRecV2Config:
    # ── Data ──
    train_tsv: str = ""
    valid_tsv: str = ""
    news_jsonl: str = ""
    semantic_dir: str = ""
    sample_pct: float = 0.10
    max_history_len: int = 128
    seq_mode: str = "A"  # A=history only, B=history+accumulated clicks

    # ── Train/Val/Test split ──
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42

    # ── Model ──
    embedding_dim: int = 384
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    codebook_size: int = 256
    code_length: int = 4

    # ── Hot News ──
    use_hot_news: bool = False
    hot_news_topk: int = 5
    hot_news_min_cat_clicks: int = 100

    # ── Training ──
    batch_size: int = 128
    lr: float = 1e-3
    codebook_lr_ratio: float = 0.1  # slow codebook updates to prevent drift
    freeze_codebook_epochs: int = 0  # 0 = learnable from step 1
    epochs: int = 50
    warmup_steps: int = 200
    eval_every: int = 2
    patience: int = 10

    # ── Scheduled Sampling ──
    use_scheduled_sampling: bool = False
    ss_warmup_epochs: int = 5     # pure TF warmup
    ss_max_prob: float = 0.3       # peak substitution probability
    ss_ramp_epochs: int = 5        # epochs to ramp from 0 → ss_max_prob
    ss_ce_floor: float = 0.3       # minimum CE weight in interpolated loss

    # ── Loss ──
    lambda_code: float = 0.25
    codebook_temperature: float = 1.0  # softmax temperature for L_code sampling

    # ── Eval ──
    beam_width: int = 50
    eval_k: list[int] = field(default_factory=lambda: [1, 5, 10])

    # ── Output ──
    output_dir: str = ""
    experiment_name: str = "genrec_v2_ablation"

    @classmethod
    def proxy(cls, **overrides: object) -> "GenRecV2Config":
        """Default proxy experiment config."""
        c = cls()
        for k, v in overrides.items():
            setattr(c, k, v)
        return c
