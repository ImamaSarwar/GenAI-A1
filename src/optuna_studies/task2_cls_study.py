"""Optuna study for the Task 2 corruption classifier. Resumable (SQLite on Drive)."""
import optuna
from src.train.train_task2_cls import run_cls_training

DRIVE_ROOT = "/content/drive/MyDrive/GenAI-A1"
TRIAL_EPOCHS = 10

SEARCH_SPACE = {
    "lr":           "loguniform [3e-4, 3e-3]",
    "batch_size":   "categorical {16, 32, 64} (multiples of 4 for exact class balance)",
    "base_ch":      "categorical {16, 32, 48, 64}",
    "dropout":      "uniform [0.0, 0.5]",
    "weight_decay": "loguniform [1e-6, 1e-2]",
}


def objective(trial):
    cfg = {
        "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "base_ch": trial.suggest_categorical("base_ch", [16, 32, 48, 64]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "epochs": TRIAL_EPOCHS,
        "run_name": f"t2cls_optuna_trial{trial.number}",
    }
    r = run_cls_training(cfg, trial=trial, experiment="task2_cls_optuna", save_ckpt=False)
    return r["best_f1"]


def get_study():
    return optuna.create_study(
        study_name="task2_classifier", direction="maximize",
        storage=f"sqlite:///{DRIVE_ROOT}/optuna/task2_cls.db", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3))
