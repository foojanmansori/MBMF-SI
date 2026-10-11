from pathlib import Path
import copy, json, hashlib
import numpy as np
import pandas as pd
import torch

import mbmf_q1_final_v7 as core
import mbmf_q1_infrastructure_v8 as pub

ROOT = Path(__file__).resolve().parents[1]


def test_scientific_core_hash_frozen():
    manifest=json.loads((ROOT/'SCIENTIFIC_CORE_MANIFEST_V8.json').read_text())
    h=hashlib.sha256((ROOT/'mbmf_q1_final_v7.py').read_bytes()).hexdigest()
    assert h==manifest['sha256']==pub.FROZEN_V7_CORE_SHA256


def test_v8_default_k_and_ablation_uniqueness():
    cfg=pub.PublicationExperimentConfigV8()
    assert cfg.core_config.latent_dims==(10,20,50)
    variants=pub.make_ablation_variants_v8(cfg)
    assert len(variants)==38
    assert len({t for t,_ in variants})==38


def test_partial_timestamps_are_disabled():
    device=torch.device('cpu')
    split=pd.DataFrame({
        'userId':[0,0,0,1,1,1],
        'movieId':[0,1,2,0,1,2],
        'rating':[3.,4.,5.,2.,3.,4.],
        'timestamp':[1.,np.nan,3.,4.,5.,6.],
        'split':['train','train','test','train','val','test'],
    })
    spec=pub.DatasetSpec('synthetic',1.,5.,True,'',1.0,'')
    cfg=pub.PublicationDataConfig(dataset='synthetic',force_backend='dense')
    data,backend=pub.build_publication_matrix_data_v8(split,spec,cfg,device)
    assert backend=='dense'
    assert data.train_timestamps is None
    assert data.train_timestamp_status=='partial_disabled'


def test_synthetic_dataset_first_class(tmp_path):
    cfg=pub.PublicationDataConfig(dataset='synthetic',data_root=str(tmp_path),synthetic_users=12,synthetic_items=15,synthetic_rank=3,synthetic_density=.4)
    df,spec=pub.load_publication_dataset(cfg)
    assert len(df)>0 and df.userId.nunique()<=12 and df.movieId.nunique()<=15
    assert spec.rating_min==1.0 and spec.rating_max==5.0


def test_bookcrossing_loader_first_class(tmp_path):
    d=tmp_path/'bookcrossing'; d.mkdir()
    pd.DataFrame({'User-ID':[1,1,2,2],'ISBN':['A','B','A','C'],'Book-Rating':[5,0,8,7]}).to_csv(d/'BX-Book-Ratings.csv',sep=';',index=False,encoding='latin-1')
    cfg=pub.PublicationDataConfig(dataset='bookcrossing',data_root=str(tmp_path))
    df,spec=pub.load_publication_dataset(cfg)
    assert set(df.rating.tolist())=={5.0,7.0,8.0}
    assert np.issubdtype(df.userId.dtype,np.integer) and np.issubdtype(df.movieId.dtype,np.integer)
    assert (spec.rating_min,spec.rating_max)==(1.0,10.0)


def test_movielens_loaders_first_class(tmp_path):
    # 100K
    d=tmp_path/'ml-100k'; d.mkdir()
    (d/'u.data').write_text('1\t2\t4\t100\n2\t3\t5\t101\n')
    df,spec=pub.load_publication_dataset(pub.PublicationDataConfig(dataset='ml-100k',data_root=str(tmp_path)))
    assert len(df)==2 and spec.rating_min==1
    # 1M
    d=tmp_path/'ml-1m'; d.mkdir()
    (d/'ratings.dat').write_text('1::2::4::100\n2::3::5::101\n')
    df,spec=pub.load_publication_dataset(pub.PublicationDataConfig(dataset='ml-1m',data_root=str(tmp_path)))
    assert len(df)==2 and spec.rating_max==5
    # 10M
    d=tmp_path/'ml-10M100K'; d.mkdir()
    (d/'ratings.dat').write_text('1::2::3.5::100\n2::3::4.5::101\n')
    df,spec=pub.load_publication_dataset(pub.PublicationDataConfig(dataset='ml-10m',data_root=str(tmp_path)))
    assert len(df)==2 and spec.rating_min==0.5


def test_jester_preprocessed_loader_first_class(tmp_path):
    d=tmp_path/'jester'; d.mkdir()
    pd.DataFrame({'userId':[1,1,2],'movieId':[0,1,0],'rating':[-2.0,5.0,1.0]}).to_csv(d/'ratings.csv',index=False)
    df,spec=pub.load_publication_dataset(pub.PublicationDataConfig(dataset='jester',data_root=str(tmp_path)))
    assert len(df)==3 and (spec.rating_min,spec.rating_max)==(-10.0,10.0)


def test_notebook_cell_ids_unique():
    import nbformat
    for name in ('mbmf_q1_final_v8.ipynb','download_datasets_q1_v8.ipynb'):
        nb=nbformat.read(ROOT/name,as_version=4)
        ids=[c.get('id') for c in nb.cells]
        assert all(ids) and len(ids)==len(set(ids))

import pytest

@pytest.mark.integration
def test_v8_strict_locked_pipeline_end_to_end(tmp_path):
    cfg=pub.PublicationExperimentConfigV8()
    cfg.data.dataset='synthetic'; cfg.data.data_root=str(tmp_path/'datasets')
    cfg.data.synthetic_users=10; cfg.data.synthetic_items=14; cfg.data.synthetic_rank=2; cfg.data.synthetic_density=.55; cfg.data.force_backend='dense'
    cfg.output_dir=str(tmp_path/'study')
    cfg.core_config.split_seeds=(1,); cfg.core_config.model_seeds=(2,); cfg.core_config.latent_dims=(4,)
    cfg.methods=tuple(f'Method{i}' for i in range(1,8))
    cfg.auto_generate_plots=False; cfg.auto_run_statistics=False; cfg.save_predictions=False; cfg.metrics.ranking_ks=(5,)
    for n in range(1,8):
        c=getattr(cfg.core_config,f'method{n}'); c.max_epochs=1; c.eval_every=1; c.val_patience=1
    candidates,locked=pub.run_publication_study_v8(cfg,device=torch.device('cpu'))
    assert not [c for c in candidates.columns if c.startswith('test_')]
    assert set(locked.method)=={f'Method{i}' for i in range(1,8)}
    assert locked.train_RMSE.notna().all() and locked.val_RMSE.notna().all() and locked.test_RMSE.notna().all()
    assert locked.selected_checkpoint.map(lambda x: Path(x).exists()).all()
