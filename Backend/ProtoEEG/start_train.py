import os
import torch
from protopnet.train_eegnet import run

# Keep this line commented if you want W&B visibility in the web UI logs section.
# os.environ["WANDB_MODE"] = "disabled"

if __name__ == "__main__":
    # These parameters are taken from the MICCAI/best_normal.yaml file Dennis sent
    run(
        backbone="spikenet_summary",
        bias_lr=0.1,
        bias_value=-1,
        cluster_coef=2.01,
        cross_entropy=8,
        num_prototypes_per_class=10,
        separation_coef=5,
        seed=0,
        detailed_logging=True,
        detailed_log_every_n_steps=1,
        detailed_log_eval=True,
        detailed_log_grad_norms=True,
        detailed_log_param_stats=True,
        collapse_std_threshold=1e-3,
        fail_on_anomaly=True,
        train_log_filename="train_log_detailed.txt",
        train_log_dir="models",
    )
