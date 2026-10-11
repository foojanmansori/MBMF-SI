from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mbmf_q1_final_v7 as v7  # noqa: E402


@pytest.mark.integration
def test_all_seven_methods_share_validation_locked_test_pipeline(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows = []
    # 10 users x 11 interactions; enough support for ranking@3 and per-user split.
    for u in range(1, 11):
        for t in range(11):
            movie = ((u + t) % 15) + 1
            rating = 0.5 + 0.5 * ((u + 2 * t) % 10)
            rows.append((u, movie, min(rating, 5.0), 100000 + 100 * u + t))
    pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"]).to_csv(data_dir / "ratings.csv", index=False)

    cfg = v7.ExperimentConfig()
    cfg.data.data_dir = str(data_dir)
    cfg.data.rating_min = 0.5
    cfg.data.rating_max = 5.0
    cfg.output_dir = str(tmp_path / "out")
    cfg.fast_debug = True
    cfg.split_seeds = (2026,)
    cfg.model_seeds = (11,)
    cfg.latent_dims = (4,)
    cfg.enable_mf_baseline = False
    cfg.enabled_baselines = ()
    cfg.enabled_methods = (1, 2, 3, 4, 5, 6, 7)
    cfg.evaluation.ranking_ks = (3,)
    cfg.evaluation.bootstrap_samples = 10
    cfg.evaluation.generate_plots = False
    cfg.evaluation.save_per_user_metrics = True
    v7.apply_q1_safe_preset(cfg)
    for mc in (cfg.method1, cfg.method2, cfg.method3, cfg.method4, cfg.method5, cfg.method6, cfg.method7):
        mc.max_epochs = 1
        mc.eval_every = 1
        mc.val_patience = 1

    candidates, locked = v7.run_all_experiments(cfg)
    assert len(candidates) == 7
    assert len(locked) == 7
    assert not [c for c in candidates if c.startswith("test_") or c.startswith("testdiag_")]
    assert {1, 2, 3, 4, 5, 6, 7} == set(cfg.enabled_methods)
    hard = locked[locked["global_rating_bound_guaranteed"].astype(bool)]
    assert locked.loc[locked.model == "Method-7-UAIBMF", "training_mode"].iloc[0] == "edge_minibatch_adamw"
    assert (hard["test_OOB_rate"] == 0).all()
    assert (hard["test_max_bound_violation"] <= 1e-8).all()
    assert (Path(cfg.output_dir) / "locked_test_summary.csv").exists()
