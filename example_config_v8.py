"""Minimal V8 programmatic configuration. Scientific V7 core remains frozen."""
import mbmf_q1_infrastructure_v8 as pub

cfg = pub.PublicationExperimentConfigV8()
cfg.data.dataset = "ml-1m"
cfg.data.data_root = "./datasets"
cfg.output_dir = "./results_q1_v8/ml-1m"

cfg.core_config.split_seeds = (2026, 2027, 2028, 2029, 2030)
cfg.core_config.model_seeds = (11, 22, 33, 44, 55)
cfg.core_config.latent_dims = (10, 20, 50)

# Engineering checks first.
pub.publication_extension_self_test_v8()

# Full locked-test study:
# candidates, selected = pub.run_publication_study_v8(cfg)

# Mechanism ablations:
# ablations = pub.run_ablation_suite_v8(cfg, "./results_q1_v8/ml-1m/ablations")

# One-factor hyperparameter scans:
# scans = pub.run_hyperparameter_scan_v8(cfg, output_root="./results_q1_v8/ml-1m/hyperparameter_scans")
