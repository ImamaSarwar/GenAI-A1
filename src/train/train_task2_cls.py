"""Task 2 classifier: balanced-batch training, macro-F1 model selection, Optuna-ready."""
import contextlib, os, time
import numpy as np, torch, torch.nn as nn
import mlflow, optuna
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

from src.data.corruptions import CLASS_NAMES
from src.data.datasets import load_pet_tensors, BalancedPetDataset, ManifestDataset
from src.data.split import load_or_make_split
from src.models.classifier import CorruptionCNN
from src.train.train_task1 import setup_mlflow

DEFAULT_CFG = dict(lr=1e-3, batch_size=32, base_ch=32, dropout=0.3, weight_decay=1e-4,
                   epochs=20, seed=42, drive_root="/content/drive/MyDrive/GenAI-A1",
                   run_name="task2_cls_baseline")


@torch.no_grad()
def evaluate_cls(model, loader, device):
    model.eval(); preds, labs = [], []
    for x, _, lab, _, _ in loader:
        preds.append(model(x.to(device)).argmax(1).cpu()); labs.append(lab)
    p, y = torch.cat(preds).numpy(), torch.cat(labs).numpy()
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=[0, 1, 2, 3], zero_division=0)
    return dict(acc=float(accuracy_score(y, p)), macro_precision=float(pr.mean()),
                macro_recall=float(rc.mean()), macro_f1=float(f1.mean()),
                per_class={n: dict(precision=float(pr[i]), recall=float(rc[i]), f1=float(f1[i]))
                           for i, n in enumerate(CLASS_NAMES)},
                cm_norm=confusion_matrix(y, p, labels=[0, 1, 2, 3], normalize="true").tolist())


def run_cls_training(cfg=None, trial=None, use_mlflow=True, experiment="task2_classifier", save_ckpt=True):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    assert cfg["batch_size"] % 4 == 0, "batch size must be a multiple of 4 for exact balance"
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainval = load_pet_tensors("trainval"); split = load_or_make_split()
    train_dl = DataLoader(BalancedPetDataset(trainval, split["train"]), batch_size=cfg["batch_size"],
                          shuffle=False, num_workers=2, drop_last=True, pin_memory=True,
                          persistent_workers=True)
    val_dl = DataLoader(ManifestDataset(trainval, "manifests/val_manifest.json"),
                        batch_size=128, shuffle=False, num_workers=2)

    model = CorruptionCNN(cfg["base_ch"], cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    crit = nn.CrossEntropyLoss()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{cfg['run_name']}: {n_params/1e6:.2f}M params | device {device}")

    ckpt_path = f"{cfg['drive_root']}/checkpoints/{cfg['run_name']}.pt"
    run_ctx = contextlib.nullcontext()
    if use_mlflow:
        setup_mlflow(cfg["drive_root"], experiment)
        run_ctx = mlflow.start_run(run_name=cfg["run_name"])

    best = dict(f1=-1, epoch=-1, res=None)
    with run_ctx:
        if use_mlflow:
            mlflow.log_params({**{k: v for k, v in cfg.items() if k != "drive_root"}, "n_params": n_params})
        for epoch in range(1, cfg["epochs"] + 1):
            t0 = time.time(); model.train(); tl, tn, tc = 0.0, 0, 0
            for x, y in train_dl:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                out = model(x); loss = crit(out, y)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                tl += loss.item() * x.size(0); tn += x.size(0); tc += (out.argmax(1) == y).sum().item()
            sched.step()
            res = evaluate_cls(model, val_dl, device)
            print(f"ep {epoch:3d} | train loss {tl/tn:.4f} acc {tc/tn:.3f} | val acc {res['acc']:.3f} "
                  f"macroF1 {res['macro_f1']:.3f} | {time.time()-t0:.1f}s")
            if use_mlflow:
                m = dict(train_loss=tl / tn, train_acc=tc / tn, val_acc=res["acc"], val_macro_f1=res["macro_f1"],
                         val_macro_precision=res["macro_precision"], val_macro_recall=res["macro_recall"])
                for n, v in res["per_class"].items(): m[f"val_f1_{n}"] = v["f1"]
                mlflow.log_metrics(m, step=epoch)
            if res["macro_f1"] > best["f1"]:
                best = dict(f1=res["macro_f1"], epoch=epoch, res=res)
                if save_ckpt:
                    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                    torch.save(dict(cfg=cfg, state_dict=model.state_dict(), epoch=epoch, val=res), ckpt_path)
            if trial is not None:
                trial.report(res["macro_f1"], epoch)
                if trial.should_prune(): raise optuna.TrialPruned()
        if use_mlflow:
            mlflow.log_metrics({"best_val_macro_f1": best["f1"], "best_epoch": best["epoch"]})
    del train_dl, val_dl
    return dict(best_f1=best["f1"], best_epoch=best["epoch"], best_res=best["res"], ckpt_path=ckpt_path)
