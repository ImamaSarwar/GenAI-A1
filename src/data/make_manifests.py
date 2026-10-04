"""
Generate the validation and test corruption manifests ONCE.
Run:  python -m src.data.make_manifests

Validation manifest : 1 entry per val image; condition drawn with equal probability,
                      severity drawn from the training ranges (continuous). severity_level = -1.
Test manifest       : 10 entries per test image: 1 clean + 3 corruptions x 3 fixed severities.
                      entry order for image i: ids i*10 + [clean, salt L/M/H, blur L/M/H, occ L/M/H]
"""
import json, os
import numpy as np

from src.data.corruptions import (CLASS_NAMES, CLEAN, SALT, BLUR, OCC,
                                  sample_params, fixed_params)
from src.data.datasets import load_pet_tensors
from src.data.split import load_or_make_split

VAL_SEED, TEST_SEED = 42, 43
OUT_DIR = "manifests"


def make_val_manifest(val_indices, seed=VAL_SEED):
    entries = []
    for eid, idx in enumerate(val_indices):
        rng = np.random.default_rng([seed, int(idx)])
        label = int(rng.integers(0, 4))
        entries.append({"id": eid, "image_idx": int(idx), "label": label,
                        "type": CLASS_NAMES[label], "severity_level": -1,
                        "params": sample_params(label, rng)})
    meta = {"split": "val", "seed": seed, "n_images": len(val_indices),
            "source": "oxford-iiit-pet trainval (image_idx = index into trainval)"}
    return {"meta": meta, "entries": entries}


def make_test_manifest(n_test, seed=TEST_SEED):
    entries = []
    for idx in range(n_test):
        entries.append({"id": len(entries), "image_idx": idx, "label": CLEAN,
                        "type": "clean", "severity_level": 0,
                        "params": {"type": "clean", "seed": 0}})
        for label in (SALT, BLUR, OCC):
            for level in range(3):
                rng = np.random.default_rng([seed, idx, label, level])
                entries.append({"id": len(entries), "image_idx": idx, "label": label,
                                "type": CLASS_NAMES[label], "severity_level": level,
                                "params": fixed_params(label, level, rng)})
    meta = {"split": "test", "seed": seed, "n_images": n_test, "entries_per_image": 10,
            "source": "oxford-iiit-pet test (image_idx = index into test)"}
    return {"meta": meta, "entries": entries}


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    split = load_or_make_split()

    val = make_val_manifest(split["val"])
    with open(f"{OUT_DIR}/val_manifest.json", "w") as f:
        json.dump(val, f)
    print("val entries :", len(val["entries"]))

    n_test = len(load_pet_tensors("test"))
    test = make_test_manifest(n_test)
    with open(f"{OUT_DIR}/test_manifest.json", "w") as f:
        json.dump(test, f)
    print("test entries:", len(test["entries"]), f"({n_test} images x 10)")
