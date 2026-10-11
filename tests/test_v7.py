from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mbmf_q1_final_v7 as v7  # noqa: E402


def tiny_data() -> v7.MatrixData:
    # Deterministic split with train support for every user and item.
    rows = []
    ratings = [1.0, 2.0, 3.0, 4.0, 5.0]
    users, items = 6, 7
    for u in range(users):
        # Four train interactions with rotating item coverage.
        for t in range(4):
            i = (u + t) % items
            rows.append((u + 1, i + 10, ratings[(u + 2 * t) % len(ratings)], 1000 + 10 * u + t, "train"))
        vi = (u + 4) % items
        ti = (u + 5) % items
        rows.append((u + 1, vi + 10, ratings[(u + 1) % len(ratings)], 2000 + u, "val"))
        rows.append((u + 1, ti + 10, ratings[(u + 3) % len(ratings)], 3000 + u, "test"))
    frame = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp", "split"])
    cfg = v7.DataConfig(rating_min=1.0, rating_max=5.0, require_explicit_rating_scale=True)
    return v7.build_matrix_data(frame, cfg, torch.device("cpu"))


def all_pairs(data: v7.MatrixData):
    u = torch.arange(data.num_users).repeat_interleave(data.num_items)
    i = torch.arange(data.num_items).repeat(data.num_users)
    return u, i


def assert_in_rating_range(pred: torch.Tensor, data: v7.MatrixData, tol: float = 1e-5):
    assert torch.isfinite(pred).all()
    assert float(pred.detach().min()) >= data.rating_min - tol
    assert float(pred.detach().max()) <= data.rating_max + tol


def test_method4_zero_residual_recovers_projected_prior_and_is_bounded():
    data = tiny_data()
    cfg = v7.Method4Config(latent_dim=5, max_abs_logit_shift=0.0)
    model = v7.ProposedMethod4(data, cfg, torch.device("cpu"))
    _, _, du, di, r_u, r_i, _, _ = model.encoded(data)
    assert torch.allclose(du, torch.zeros_like(du))
    assert torch.allclose(di, torch.zeros_like(di))
    assert torch.allclose(r_u, model.rW0, atol=1e-6, rtol=1e-6)
    assert torch.allclose(r_i, model.rH0, atol=1e-6, rtol=1e-6)
    assert float(r_u.max()) < math.sqrt(data.half_range)
    assert float(r_i.max()) < math.sqrt(data.half_range)
    u, i = all_pairs(data)
    assert_in_rating_range(model.predict_raw_pairs(data, u, i), data)


def test_method4_extreme_residual_head_cannot_break_rating_bound():
    data = tiny_data()
    cfg = v7.Method4Config(latent_dim=6, max_abs_logit_shift=2.0, residual_transform="smooth_tanh")
    model = v7.ProposedMethod4(data, cfg, torch.device("cpu"))
    with torch.no_grad():
        model.user_branch.magnitude_head.weight.fill_(1e3)
        model.user_branch.magnitude_head.bias.fill_(1e3)
        model.item_branch.magnitude_head.weight.fill_(-1e3)
        model.item_branch.magnitude_head.bias.fill_(-1e3)
    u, i = all_pairs(data)
    assert_in_rating_range(model.predict_raw_pairs(data, u, i), data)
    diag = model.diagnostics(data, u, i)
    assert diag["max_rating_bound_violation"] <= 1e-6
    assert diag["max_structural_bound_violation"] <= 1e-5


def test_method2_squared_loss_decomposition_is_exact():
    torch.manual_seed(0)
    y = torch.randn(31)
    pu = torch.randn(31)
    pi = torch.randn(31)
    gamma = 0.37
    fused = gamma * pu + (1.0 - gamma) * pi
    lhs = gamma * torch.mean((pu - y) ** 2) + (1.0 - gamma) * torch.mean((pi - y) ** 2)
    rhs = torch.mean((fused - y) ** 2) + gamma * (1.0 - gamma) * torch.mean((pu - pi) ** 2)
    assert torch.allclose(lhs, rhs, atol=1e-6, rtol=1e-6)



def test_sparse_dv_encoder_matches_dense_values_plus_mask_linear():
    data = tiny_data()
    width = data.rating_max - data.rating_min
    values = ((data.train_ratings_raw - data.rating_min) / width).clamp(0.0, 1.0)
    enc = v7.SparseHistoryEncoderV7(
        data.train_users, data.train_items, values, data.num_users, data.num_items,
        hidden_dim=9, latent_dim=4, input_mode="values_plus_mask",
    )
    actual = enc.forward_all()
    x = torch.zeros(data.num_users, data.num_items)
    m = torch.zeros_like(x)
    x[data.train_users, data.train_items] = values
    m[data.train_users, data.train_items] = 1.0
    hidden = torch.relu(x @ enc.value_weight + m @ enc.mask_weight + enc.input_bias)
    expected = enc.output(hidden)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)

def test_method6_range_safe_all_pairs_and_fixed_radius_ablation():
    data = tiny_data()
    for mode in ("range_safe", "fixed"):
        cfg = v7.Method6Config(latent_dim=5, hidden_dim=12, radius_mode=mode)
        model = v7.DecoderCoupledDualViewMBMF(data, cfg, torch.device("cpu"))
        u, i = all_pairs(data)
        assert_in_rating_range(model.predict_raw_pairs(data, u, i), data)
        diag = model.diagnostics(data, u, i)
        assert diag["max_structural_bound_violation"] <= 1e-5


def test_method7_is_bounded_in_train_and_eval_modes():
    data = tiny_data()
    cfg = v7.Method7Config(latent_dim=5, hidden_dim=10, feature_dropout=0.4)
    model = v7.ProposedMethod7(data, cfg, torch.device("cpu"))
    u, i = all_pairs(data)
    model.train()
    for _ in range(3):
        assert_in_rating_range(model.predict_raw_pairs(data, u, i), data)
    model.eval()
    assert_in_rating_range(model.predict_raw_pairs(data, u, i), data)


def test_method7_homoscedastic_uses_only_global_variance_parameter():
    data = tiny_data()
    cfg = v7.Method7Config(latent_dim=4, hidden_dim=8, heteroscedastic=False, consistency_weight=0.0)
    model = v7.ProposedMethod7(data, cfg, torch.device("cpu"))
    model.train()
    loss, _ = model.loss(data)
    loss.backward()
    assert model.log_variance_prior.grad is not None
    assert torch.isfinite(model.log_variance_prior.grad)
    for head in (model.user_logvariance, model.item_logvariance):
        assert all(p.grad is None for p in head.parameters())



def test_method7_uncertainty_metrics_are_explicit_and_mse_ablation_suppresses_them():
    data = tiny_data()
    robust = v7.ProposedMethod7(data, v7.Method7Config(latent_dim=4, hidden_dim=8, feature_dropout=0.0), torch.device("cpu"))
    metrics = v7.evaluate_model(robust, data, "val")
    for key in ("Gaussian_NLL", "PICP50", "PICP80", "PICP90", "PICP95", "coverage_calibration_MAE"):
        assert key in metrics and np.isfinite(metrics[key]), key
    mse_cfg = v7.Method7Config(latent_dim=4, hidden_dim=8, feature_dropout=0.0, loss_mode="mse", consistency_weight=0.0)
    mse_model = v7.ProposedMethod7(data, mse_cfg, torch.device("cpu"))
    mse_metrics = v7.evaluate_model(mse_model, data, "val")
    assert "Gaussian_NLL" not in mse_metrics
    assert "PICP90" not in mse_metrics

def test_prediction_cache_matches_direct_predictions_for_methods_1_to_7():
    data = tiny_data()
    device = torch.device("cpu")
    models = [
        v7.ThesisMethod1(data, v7.Method1Config(latent_dim=4, hidden_dim=8), device),
        v7.ThesisMethod2(data, v7.Method2Config(latent_dim=4), device),
        v7.ProposedMethod3(data, v7.Method3Config(latent_dim=4), device),
        v7.ProposedMethod4(data, v7.Method4Config(latent_dim=4), device),
        v7.DecoderCoupledDualViewMBMF(data, v7.Method5Config(latent_dim=4, hidden_dim=8), device),
        v7.DecoderCoupledDualViewMBMF(data, v7.Method6Config(latent_dim=4, hidden_dim=8), device),
        v7.ProposedMethod7(data, v7.Method7Config(latent_dim=4, hidden_dim=8, feature_dropout=0.0), device),
    ]
    u, i = all_pairs(data)
    for model in models:
        model.eval()
        direct = v7._predict_raw_generic(model, data, u, i)
        cache = v7.build_eval_prediction_cache(model, data)
        cached = v7._predict_raw_cached(cache, data, u, i)
        assert torch.allclose(direct, cached, atol=1e-6, rtol=1e-6), type(model).__name__



def test_hierarchical_bootstrap_uses_outer_splits_and_user_pairing():
    rows = []
    for ss in (1, 2, 3):
        for seed in (11, 22):
            for u in range(5):
                rows.append({"split_seed": ss, "model_seed": seed, "model": "REF", "user": u, "user_MSE": 1.0 + 0.01 * u})
                rows.append({"split_seed": ss, "model_seed": seed, "model": "ALT", "user": u, "user_MSE": 0.8 + 0.01 * u})
    frame = pd.DataFrame(rows)
    out = v7.hierarchical_paired_bootstrap_vs_reference(frame, "REF", samples=200, seed=7)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["outer_splits"] == 3
    assert row["paired_users_total"] == 15
    assert row["delta_user_MSE"] < 0
    assert row["ci95_high"] < 0

def test_ablation_registry_is_isolated_and_complete():
    cfg = v7.ExperimentConfig()
    v7.apply_q1_safe_preset(cfg)
    variants = v7.make_ablation_variants(cfg)
    assert len(variants) == 36
    tags = [x[0] for x in variants]
    assert len(set(tags)) == len(tags)
    for tag, acfg in variants:
        assert acfg.enable_mf_baseline is False, tag
        assert acfg.enabled_baselines == (), tag
        assert len(acfg.enabled_methods) == 1, tag


def test_legacy_direct_candidate_helpers_are_guarded():
    for name in ("run_paper_once", "run_method1_once", "run_method2_once", "run_method3_once"):
        fn = getattr(v7, name)
        with pytest.raises(RuntimeError):
            fn(None)
