"""Infrastructure-only regressions: do NOT alter Methods 1-7 or the ablation plan."""
from pathlib import Path
import copy

import numpy as np
import pandas as pd
import pytest
import torch

import mbmf_q1_final_v7 as core
import mbmf_q1_infrastructure_v8 as pub


def _tiny(cfg, device=torch.device('cpu')):
    df, spec = pub.load_publication_dataset(cfg.data)
    split = pub.make_core_split(df, spec, cfg.data)
    data, _ = pub.build_publication_matrix_data_v8(split, spec, cfg.data, device)
    return data


def _config(tmp_path, backend='dense', signed=False):
    cfg = pub.PublicationExperimentConfigV8()
    cfg.data.dataset = 'synthetic'
    cfg.data.data_root = str(tmp_path)
    cfg.data.synthetic_users = 11
    cfg.data.synthetic_items = 15
    cfg.data.synthetic_density = .72
    cfg.data.synthetic_rank = 3
    cfg.data.force_backend = backend
    if signed:
        cfg.data.synthetic_rating_min = -10.0
        cfg.data.synthetic_rating_max = 10.0
    cfg.core_config.latent_dims = (3,)
    cfg.core_config.split_seeds = (21,)
    cfg.core_config.model_seeds = (5,)
    cfg.save_predictions = False
    cfg.auto_generate_plots = False
    cfg.auto_run_statistics = False
    cfg.output_dir = str(tmp_path / 'study')
    return cfg


@pytest.mark.parametrize('backend', ['dense', 'sparse'])
def test_jester_style_signed_mbmf_c_n_checkpoint_roundtrip(tmp_path, backend):
    cfg = _config(tmp_path, backend, signed=True)
    data = _tiny(cfg)
    for method in ('MBMF-F', 'MBMF-C', 'MBMF-N'):
        model, train_cfg = pub.instantiate_model_v8(method, data, cfg, 3, 17,
                                                    torch.device('cpu'), 'isotropic_gaussian')
        train_cfg.max_epochs, train_cfg.eval_every = 2, 1
        history = pub.train_model_v8(method, model, data, train_cfg, cfg)
        assert history['elapsed_sec'] >= 0
        if method != 'MBMF-F':
            assert isinstance(model, pub.OriginShiftedPaperMBMF)
        else:
            assert type(model) is core.PaperMBMF
        pred = pub._predict_raw_chunked(model, data, data.test_users, data.test_items, 4)
        assert torch.isfinite(pred).all()
        lo, hi = pub.individual_bound_arrays_v8(model, data, data.test_users, data.test_items)
        assert np.all(pred.detach().numpy() >= lo - 1e-4)
        assert np.all(pred.detach().numpy() <= hi + 1e-4)
        ck = tmp_path / f'{method}-{backend}.pt'
        pub.save_model_checkpoint_v8(ck, model, config=train_cfg, data=data,
                                     extra={'init_mode': 'isotropic_gaussian'})
        restored, payload = pub.load_model_checkpoint_v8(ck, data, torch.device('cpu'))
        pred2 = pub._predict_raw_chunked(restored, data, data.test_users, data.test_items, 4)
        assert torch.allclose(pred, pred2, atol=1e-6)
        assert payload['frozen_v7_core_sha256'] == pub.FROZEN_V7_CORE_SHA256


def test_method7_gaussian_calibration_and_eval(tmp_path):
    mean = np.array([0., 0., 0., 0.])
    truth = np.array([0., 2., -2., 0.])
    variance = np.ones(4)
    actual = pub.gaussian_uncertainty_metrics_v8(mean, truth, variance, 4.)
    assert actual['Gaussian_NLL'] == pytest.approx(np.mean(.5*np.log(2*np.pi)+.5*truth**2))
    assert actual['PICP_50'] == .5
    assert actual['PICP_95'] == .5
    assert actual['MPIW_90'] == pytest.approx(2*1.6448536269514722)
    assert actual['NormalizedMPIW_90'] == pytest.approx(actual['MPIW_90']/4.)
    cfg = _config(tmp_path)
    data = _tiny(cfg)
    model, _ = pub.instantiate_model_v8('Method7', data, cfg, 3, 5, torch.device('cpu'))
    metrics = pub.evaluate_publication_model(model, data, 'test', chunk_size=4)
    for pct in (50, 80, 90, 95):
        for prefix in ('PICP', 'CoverageGap', 'MPIW', 'NormalizedMPIW'):
            assert np.isfinite(metrics[f'{prefix}_{pct}'])
    assert np.isfinite(metrics['Gaussian_NLL'])


def test_ranking_and_f1_unchanged_by_linear_grouping():
    users = np.array([0, 2, 1, 2, 0, 1, 2, 0, 2, 1], dtype=np.int64)
    true = np.array([4., 2., 3., 4., 1., 4., 5., 2., 3., 5.])
    pred = np.array([3., 3., 3., 4., 1., 4., 2., 4., 4., 5.])
    means = np.array([2.5, 3.0, 3.5])
    a = pub.train_threshold_f1_fast_v8(users, true, pred, means)
    b = core.train_threshold_f1(torch.tensor(users), torch.tensor(true), torch.tensor(pred), torch.tensor(means))
    for k in a:
        assert a[k] == pytest.approx(b[k], abs=1e-6)
    got = pub.testset_ranking_metrics(users, true, pred, (1, 2, 5), means, rating_min=1.)
    # Slow original/reference calculation deliberately independent of grouping optimization.
    baseline = {}
    for k in (1, 2, 5):
        vals = []
        for u in np.unique(users):
            idx = np.flatnonzero(users == u)
            yt = true[idx]
            pp = pred[idx]
            order = np.argsort(-pp, kind='stable')
            ideal = np.argsort(-yt, kind='stable')
            def dcg(x):
                return float(np.sum(x[:min(k, len(x))] / np.log2(np.arange(2, 2+min(k, len(x))))))
            vals.append(dcg(yt[order])/dcg(yt[ideal]))
        baseline[f'NDCG_testset@{k}'] = float(np.mean(vals))
    for k in baseline:
        assert got[k] == pytest.approx(baseline[k], abs=1e-12)


def test_external_predictions_must_match_exact_split(tmp_path):
    cfg = _config(tmp_path)
    data = _tiny(cfg)
    exported = pub.export_split_for_external(data, tmp_path/'common.csv')
    df = pd.read_csv(exported)
    assert {'split','userId','movieId','rating'}.issubset(df.columns)
    assert not any(c.startswith('original_') for c in df.columns)  # no misleading originals
    df['prediction'] = df.rating
    fp = tmp_path / 'pred.csv'
    df.to_csv(fp, index=False)
    row = pub.evaluate_external_predictions_v8(data, fp, 'external')
    assert row['test_RMSE'] == 0
    df.iloc[:-1].to_csv(fp, index=False)
    with pytest.raises(ValueError, match='exact'):
        pub.evaluate_external_predictions_v8(data, fp, 'external')
    df = pd.read_csv(exported)
    df['prediction'] = df.rating
    pd.concat([df, df.iloc[[0]]]).to_csv(fp, index=False)
    with pytest.raises(ValueError, match='duplicated'):
        pub.evaluate_external_predictions_v8(data, fp, 'external')


def test_external_split_original_ids_are_not_relabelled_as_internal(tmp_path):
    cfg = _config(tmp_path)
    raw, spec = pub.load_publication_dataset(cfg.data, save_mappings_to=tmp_path/'maps')
    split = pub.make_core_split(raw, spec, cfg.data)
    data,_ = pub.build_publication_matrix_data_v8(split, spec, cfg.data, torch.device('cpu'))
    df = pd.read_csv(pub.export_split_for_external(data, tmp_path/'mapped.csv',
                 id_map_dir=tmp_path/'maps', dataset_name='synthetic'))
    assert 'original_user_id' in df.columns
    map_df = pd.read_csv(tmp_path/'maps'/'synthetic_user_id_map.csv')
    lookup = dict(zip(map_df.userId, map_df.original_user_id))
    assert all(int(o) == lookup[u] for u, o in zip(df.userId, df.original_user_id))


def test_hyperparameter_scan_is_validation_only(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    cfg.methods = ('Method1',)
    cfg.core_config.method1.max_epochs = 1
    cfg.core_config.method1.eval_every = 1
    cfg.core_config.method1.val_patience = 1
    jobs=[]
    for x in (.5, .7):
        c=copy.deepcopy(cfg)
        c.core_config.method1.alpha_ae = x
        jobs.append((f'M1_alpha={x}', c))
    monkeypatch.setattr(pub, 'make_hyperparameter_scan_jobs_v8', lambda base, plan=None: jobs)
    real_eval=pub.evaluate_publication_model
    called=[]
    def guard(model,data,split,*args,**kw):
        if split == 'test':
            raise AssertionError('test evaluated during hyperparameter scan')
        called.append(split)
        return real_eval(model,data,split,*args,**kw)
    monkeypatch.setattr(pub, 'evaluate_publication_model', guard)
    table = pub.run_hyperparameter_scan_v8(cfg,output_root=tmp_path/'scan',device=torch.device('cpu'))
    assert set(table.scan) == {'M1_alpha=0.5','M1_alpha=0.7'}
    assert 'val_RMSE' in table
    assert not any(c.startswith('test_') for c in table.columns)
    assert (tmp_path/'scan'/'hyperparameter_scan_validation_runs.csv').exists()
    assert called and set(called).issubset({'train','val'})


def test_frozen_scientific_core_preserved():
    assert pub.sha256_file(pub.V7_CORE_PATH) == '6da726cb155ab6ad291f5f608febc37f42d1c25ac3924910f56917908c1f124c'
    assert len(pub.make_ablation_variants_v8(pub.PublicationExperimentConfigV8())) == 38
    assert tuple(pub.DATASET_SPECS) == ('synthetic','jester','bookcrossing','ml-100k','ml-1m','ml-10m')


def test_scan_and_ablation_summaries_not_pooled():
    runs = pd.DataFrame([
        {'dataset':'synthetic', 'method':'Method1','K':10,'init_mode':'NA','scan':'alpha=.5','test_RMSE':1.},
        {'dataset':'synthetic', 'method':'Method1','K':10,'init_mode':'NA','scan':'alpha=.7','test_RMSE':3.},
    ])
    summ = pub.summarise_publication_runs(runs)
    assert len(summ) == 2
    assert set(summ['test_RMSE_mean']) == {1.,3.}
    runs.rename(columns={'scan':'ablation'},inplace=True)
    assert len(pub.summarise_publication_runs(runs)) == 2


def test_friedman_uses_only_common_methods_across_dataset_blocks():
    rows = []
    for ds in ('synthetic','ml-100k'):
        for seed in (1,2,3):
            for method, score in [('Method1',1.0),('Method2',2.0),('Method3',1.5)]:
                rows.append({'dataset':ds, 'split_seed':seed, 'method':method,
                             'test_RMSE':score+seed/20})
            if ds=='ml-100k':
                rows.append({'dataset':ds,'split_seed':seed,'method':'AutoRec', 'test_RMSE':.3})
    ranks, stats = pub.friedman_nemenyi_summary(pd.DataFrame(rows))
    assert set(ranks.method) == {'Method1','Method2','Method3'}
    assert stats['N']==6 and stats['K']==3
    assert 'AutoRec' not in stats['methods_included']


@pytest.mark.parametrize('variant',['MBMF-F','MBMF-C','MBMF-N'])
def test_paper_dense_sparse_single_step_match(tmp_path, variant):
    cfg=_config(tmp_path)
    raw,spec=pub.load_publication_dataset(cfg.data)
    split=pub.make_core_split(raw,spec,cfg.data)
    dense_cfg=copy.deepcopy(cfg.data); dense_cfg.force_backend='dense'
    sparse_cfg=copy.deepcopy(cfg.data); sparse_cfg.force_backend='sparse'
    dense,_=pub.build_publication_matrix_data_v8(split,spec,dense_cfg,torch.device('cpu'))
    sparse,_=pub.build_publication_matrix_data_v8(split,spec,sparse_cfg,torch.device('cpu'))
    c=copy.deepcopy(cfg.core_config.paper)
    c.latent_dim=3; c.max_epochs=1; c.eval_every=1
    torch.manual_seed(123)
    a=core.PaperMBMF(dense,c,variant,'isotropic_gaussian',torch.device('cpu'))
    torch.manual_seed(123)
    b=core.PaperMBMF(sparse,c,variant,'isotropic_gaussian',torch.device('cpu'))
    core.train_paper_mbmf(a,dense,c)
    pub.train_paper_mbmf_chunked(b,sparse,c,11)
    for key in a.state_dict():
        assert torch.allclose(a.state_dict()[key], b.state_dict()[key],atol=1e-5,rtol=1e-5),key
