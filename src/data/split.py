"""80/20 split of the Oxford-IIIT Pet *trainval* set with seed 42.
Indices are saved to manifests/split_seed42.json so every task uses the same split."""
import json, os
import torch

SPLIT_PATH = "manifests/split_seed42.json"


def make_split(n_total=3680, val_frac=0.2, seed=42):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=g).tolist()
    n_val = int(round(n_total * val_frac))
    return {"seed": seed, "n_total": n_total,
            "val": sorted(perm[:n_val]), "train": sorted(perm[n_val:])}


def load_or_make_split(path=SPLIT_PATH, n_total=3680):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    split = make_split(n_total)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(split, f)
    return split


if __name__ == "__main__":
    s = load_or_make_split()
    print(f"train={len(s['train'])} val={len(s['val'])}")
