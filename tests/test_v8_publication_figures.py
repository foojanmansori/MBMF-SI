"""Visualization/regression tests; don't train models or modify experimental design."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from mbmf_q1_visuals_v8 import (method_style, generate_publication_figures_v8,
                                   generate_ablation_figures_v8, generate_multi_dataset_figures_v8)
import mbmf_q1_infrastructure_v8 as pub


def test_14_distinct_fixed_colors_and_shapes():
    names=['MF','BiasMF','MBMF-F','MBMF-C','MBMF-N','AutoRec','AEMC-NE']+[f'Method{i}' for i in range(1,8)]
    styles=[method_style(x) for x in names]
    assert len({c for c,m in styles})==14
    assert styles == [method_style(x) for x in names]


def _locked(tmp_path, names=('MF','Method1','Method7')):
    out=[]
    for j,m in enumerate(names):
        file=tmp_path/(m+'.npz')
        t=np.array([1,2,3,4,5]*3,dtype='float32')
        np.savez_compressed(file,truth=t,prediction=t+np.float32(.1*(j+1)))
        out.append(dict(dataset='ml-100k',method=m,split_seed=2026,model_seed=11,K=10,
            test_prediction_file=str(file),test_RMSE=.7+j*.05,test_MAE=.56+j*.04,
            test_Global_OOB_Rate=0.0,train_RMSE=.4+j*.03,val_RMSE=.6+j*.04,
            train_wall_sec=1.0+j,test_NDCG_testset__1=.4+j*.05))
    return pd.DataFrame(out)


def test_presentation_does_not_change_metrics_and_no_fake_ci(tmp_path):
    locked=_locked(tmp_path)
    before=locked.copy(deep=True)
    r=generate_publication_figures_v8(locked,tmp_path)
    pd.testing.assert_frame_equal(locked,before)
    assert len(r['figures'])>=7
    assert 'NOT inferentially valid' in r['statistical_note']
    assert (tmp_path/'figure_manifest.json').exists()
    assert (tmp_path/'figures'/'q1_rank_test_RMSE.pdf').exists()
    assert (tmp_path/'figures'/'q1_absolute_error_ecdf.png').exists()
    assert (tmp_path/'figures'/'q1_error_by_rating.pdf').exists()
    for file in tmp_path.glob('figures/*.pdf'):
        assert file.stat().st_size>500


def test_ablation_plot_one_panel_per_family(tmp_path):
    df=pd.DataFrame([dict(ablation='m1_full',method='Method1',test_RMSE=.80),
                     dict(ablation='m1_remove_ae',method='Method1',test_RMSE=.89),
                     dict(ablation='m2_full',method='Method2',test_RMSE=.85),
                     dict(ablation='m2_remove_ae',method='Method2',test_RMSE=.99)])
    res=generate_ablation_figures_v8(df,tmp_path)
    assert len(res)==2
    assert (tmp_path/'figures'/'q1_ablation_delta_Method1.pdf').exists()


def test_cross_dataset_heatmap(tmp_path):
    a=_locked(tmp_path)[['dataset','method','test_RMSE']]
    b=a.copy();b['dataset']='ml-1m';b['test_RMSE']+=.1
    r=generate_multi_dataset_figures_v8(pd.concat([a,b],ignore_index=True),tmp_path)
    assert len(r)==1
    assert (tmp_path/'figures'/'q1_cross_dataset_rmse_heatmap.pdf').exists()


def test_summary_preserves_grouping_and_counts():
    d=pd.DataFrame([{'dataset':'s','method':'M1','K':10,'init_mode':'NA','test_RMSE':1.,'split_seed':1,'model_seed':3},
                    {'dataset':'s','method':'M1','K':10,'init_mode':'NA','test_RMSE':2.,'split_seed':1,'model_seed':4},
                    {'dataset':'s','method':'M1','K':20,'init_mode':'NA','test_RMSE':3.,'split_seed':1,'model_seed':3}])
    x=pub.summarise_publication_runs(d)
    assert len(x)==2
    assert x.loc[x.K==10,'test_RMSE_mean'].iloc[0]==pytest.approx(1.5)
    assert x.loc[x.K==10,'test_RMSE_count'].iloc[0]==2


def test_immutability_compares_to_repository_pinned_manifest(tmp_path, monkeypatch):
    import hashlib
    fake = tmp_path / 'edited_v7.py'
    fake.write_text('# local scientific core was accidentally modified\n', encoding='utf-8')
    monkeypatch.setattr(pub, 'V7_CORE_PATH', fake)
    # Recreate the *previous* false-positive: import-time SHA matches modified
    # file, but committed independent manifest must now catch it.
    monkeypatch.setattr(pub, 'FROZEN_V7_CORE_SHA256', hashlib.sha256(fake.read_bytes()).hexdigest())
    with pytest.raises(AssertionError, match='GitHub-pinned'):
        pub.scientific_core_immutability_test_v8()
