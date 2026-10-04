"""
Task 1 training / evaluation. Used for the baseline run now and by Optuna later.
    from src.train.train_task1 import run_training
    result = run_training({"lr": 1e-3, "epochs": 40})
"""
import contextlib, os, time
import numpy as np
import torch
import mlflow, optuna
from torch.utils.data import DataLoader

from src.data.corruptions import CLASS_NAMES
from src.data.datasets import load_pet_tensors, PetTrainDataset, ManifestDataset
from src.data.split import load_or_make_split
from src.losses.losses import recon_loss, psnr_per_sample, ssim_per_sample
from src.models.autoencoder import DAE, ConvDAE, build_model

DEFAULT_CFG = dict(lr=1e-3, batch_size=32, arch="conv", base_ch=32, latent_dim=256, latent_ch=16, dropout=0.1,
                   alpha=0.8, weight_decay=1e-5, epochs=40, seed=42,
                   drive_root="/content/drive/MyDrive/GenAI-A1", run_name="task1_baseline")


def setup_mlflow(drive_root, experiment):
    mlflow.set_tracking_uri(f"sqlite:///{drive_root}/mlflow.db")
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=f"{drive_root}/mlruns")
    mlflow.set_experiment(experiment)


@torch.no_grad()
def evaluate(model, loader, device, alpha=0.8):
    """Per-sample metrics over a ManifestDataset loader. Returns dict with 'overall' and per-class stats."""
    model.eval()
    labs, sev, o_psnr, o_ssim, o_l1, i_psnr, i_ssim = [], [], [], [], [], [], []
    for x, y, lab, lvl, _ in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        o_psnr.append(psnr_per_sample(out, y).cpu()); o_ssim.append(ssim_per_sample(out, y).cpu())
        o_l1.append((out - y).abs().mean(dim=(1, 2, 3)).cpu())
        i_psnr.append(psnr_per_sample(x, y).cpu());   i_ssim.append(ssim_per_sample(x, y).cpu())
        labs.append(lab); sev.append(lvl)
    cat = lambda l: torch.cat(l).numpy()
    labs, sev = cat(labs), cat(sev)
    d = dict(psnr=cat(o_psnr), ssim=cat(o_ssim), l1=cat(o_l1), in_psnr=cat(i_psnr), in_ssim=cat(i_ssim))
    res = {"overall": {k: float(v.mean()) for k, v in d.items()}, "per_class": {}, "per_class_severity": {}}
    res["overall"]["loss"] = alpha * res["overall"]["l1"] + (1 - alpha) * (1 - res["overall"]["ssim"])
    for c, name in enumerate(CLASS_NAMES):
        m = labs == c
        if m.any():
            res["per_class"][name] = {k: float(v[m].mean()) for k, v in d.items()}
            for s in np.unique(sev[m]):
                ms = m & (sev == s)
                res["per_class_severity"][f"{name}/{int(s)}"] = {k: float(v[ms].mean()) for k, v in d.items()}
    # model-selection score: restoration quality only (independent of alpha, so alpha can't "game" it)
    res["score"] = res["overall"]["ssim"] + res["overall"]["psnr"] / 50.0
    return res


def run_training(cfg=None, trial=None, use_mlflow=True, max_train=None,
                 experiment="task1_universal_dae", save_ckpt=True):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainval = load_pet_tensors("trainval")
    split = load_or_make_split()
    train_idx = split["train"][:max_train] if max_train else split["train"]
    train_dl = DataLoader(PetTrainDataset(trainval, train_idx), batch_size=cfg["batch_size"],
                          shuffle=True, num_workers=2, drop_last=True, pin_memory=True,
                          persistent_workers=True)
    val_dl = DataLoader(ManifestDataset(trainval, "manifests/val_manifest.json"),
                        batch_size=128, shuffle=False, num_workers=2)

    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{cfg['run_name']}: {n_params/1e6:.2f}M params | train imgs {len(train_idx)} | device {device}")

    ckpt_path = f"{cfg['drive_root']}/checkpoints/{cfg['run_name']}.pt"
    run_ctx = contextlib.nullcontext()
    if use_mlflow:
        setup_mlflow(cfg["drive_root"], experiment)
        run_ctx = mlflow.start_run(run_name=cfg["run_name"])

    best = dict(score=-1, epoch=-1, res=None)
    with run_ctx:
        if use_mlflow:
            mlflow.log_params({**{k: v for k, v in cfg.items() if k != "drive_root"}, "n_params": n_params})
        for epoch in range(1, cfg["epochs"] + 1):
            t0 = time.time(); model.train(); tl, tn = 0.0, 0
            for x, y, _ in train_dl:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                loss, _, _ = recon_loss(model(x), y, cfg["alpha"])
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                tl += loss.item() * x.size(0); tn += x.size(0)
            sched.step()
            res = evaluate(model, val_dl, device, cfg["alpha"])
            o = res["overall"]
            print(f"ep {epoch:3d} | train {tl/tn:.4f} | val loss {o['loss']:.4f} PSNR {o['psnr']:.2f} "
                  f"SSIM {o['ssim']:.4f} | {time.time()-t0:.1f}s")
            if use_mlflow:
                m = {"train_loss": tl / tn, "val_loss": o["loss"], "val_psnr": o["psnr"],
                     "val_ssim": o["ssim"], "val_l1": o["l1"], "val_score": res["score"],
                     "lr": opt.param_groups[0]["lr"]}
                for cn, v in res["per_class"].items():
                    m[f"val_psnr_{cn}"] = v["psnr"]; m[f"val_ssim_{cn}"] = v["ssim"]
                mlflow.log_metrics(m, step=epoch)
            if res["score"] > best["score"]:
                best = dict(score=res["score"], epoch=epoch, res=res)
                if save_ckpt:
                    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                    torch.save(dict(cfg=cfg, state_dict=model.state_dict(), epoch=epoch,
                                    val=res), ckpt_path)
            if trial is not None:                      # Optuna pruning hook
                trial.report(res["score"], epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        if use_mlflow:
            mlflow.log_metrics({"best_val_score": best["score"], "best_epoch": best["epoch"]})
    del train_dl, val_dl
    return dict(best_score=best["score"], best_epoch=best["epoch"], best_res=best["res"],
                ckpt_path=ckpt_path, n_params=n_params)
