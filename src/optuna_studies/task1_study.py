"""Optuna study for Task 1. Resumable: storage is SQLite on Drive, load_if_exists=True."""
import optuna
from src.train.train_task1 import run_training

DRIVE_ROOT = "/content/drive/MyDrive/GenAI-A1"
STUDY_NAME = "task1_universal_dae"
TRIAL_EPOCHS = 12

SEARCH_SPACE = {
    "lr":         "loguniform [3e-4, 3e-3]",
    "batch_size": "categorical {16, 32, 64}",
    "base_ch":    "categorical {16, 32, 48, 64}",
    "latent_ch":  "categorical {8, 16, 32, 64}  (8x8 spatial latent -> 512..4096 values, >=12x compression)",
    "dropout":    "uniform [0.0, 0.3]",
    "alpha":      "uniform [0.5, 0.95]  (L1 weight in alpha*L1 + (1-alpha)*(1-SSIM))",
}


def objective(trial):
    cfg = {
        "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "base_ch": trial.suggest_categorical("base_ch", [16, 32, 48, 64]),
        "latent_ch": trial.suggest_categorical("latent_ch", [8, 16, 32, 64]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "alpha": trial.suggest_float("alpha", 0.5, 0.95),
        "epochs": TRIAL_EPOCHS,
        "run_name": f"t1_optuna_trial{trial.number}",
    }
    r = run_training(cfg, trial=trial, experiment="task1_optuna", save_ckpt=False)
    return r["best_score"]          # SSIM + PSNR/50, independent of alpha


def get_study():
    return optuna.create_study(
        study_name=STUDY_NAME, direction="maximize",
        storage=f"sqlite:///{DRIVE_ROOT}/optuna/task1.db", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=4))
