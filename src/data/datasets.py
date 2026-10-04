"""
Datasets for Tasks 1-3.

- load_pet_tensors : loads Oxford-IIIT Pet once, converts to RGB, resizes to 128x128,
                     caches as a uint8 tensor (N,3,128,128) -> fast, no disk reads in training.
- PetTrainDataset  : corruption sampled at RUNTIME every time an image is loaded.
- ManifestDataset  : deterministic corruption read from a manifest (validation / test).
"""
import json, os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet
from tqdm.auto import tqdm

from src.data.corruptions import (IMG_SIZE, apply_corruption, sample_params)


def load_pet_tensors(split, root="/content/data", size=IMG_SIZE):
    """split: 'trainval' or 'test'. Returns uint8 tensor (N, 3, size, size)."""
    cache = os.path.join(root, f"pets_{split}_{size}.pt")
    if os.path.exists(cache):
        return torch.load(cache)
    ds = OxfordIIITPet(root=root, split=split, download=True)
    arr = torch.empty(len(ds), 3, size, size, dtype=torch.uint8)
    for i in tqdm(range(len(ds)), desc=f"loading {split}"):
        img, _ = ds[i]
        img = img.convert("RGB").resize((size, size), Image.BILINEAR)
        arr[i] = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1)
    os.makedirs(root, exist_ok=True)
    torch.save(arr, cache)
    return arr


class PetTrainDataset(Dataset):
    """
    Returns (corrupted, clean, label)  [+ params JSON string if return_meta=True].
    A NEW corruption type + severity is sampled on every __getitem__ call.

    conditions: which labels may be sampled (equal probability among them).
        (0,1,2,3) -> universal model / classifier (Task 1, 2)
        (1,)      -> salt specialist only (Task 2); (2,) blur; (3,) occlusion
    """
    def __init__(self, images, indices, conditions=(0, 1, 2, 3), return_meta=False):
        self.images = images
        self.indices = list(indices)
        self.conditions = list(conditions)
        self.return_meta = return_meta
        self._rng, self._pid = None, None

    def __len__(self):
        return len(self.indices)

    def _get_rng(self):
        pid = os.getpid()                       # separate RNG stream per DataLoader worker
        if self._rng is None or self._pid != pid:
            self._rng = np.random.default_rng([torch.initial_seed() % 2**32, pid])
            self._pid = pid
        return self._rng

    def __getitem__(self, i):
        rng = self._get_rng()
        clean = self.images[self.indices[i]].float() / 255.0
        label = int(rng.choice(self.conditions))
        params = sample_params(label, rng)
        corrupted = apply_corruption(clean, params)
        if self.return_meta:
            return corrupted, clean, label, json.dumps(params)
        return corrupted, clean, label


class ManifestDataset(Dataset):
    """
    Deterministic dataset driven by a manifest file.
    Returns (corrupted, clean, label, severity_level, entry_id).
    `images` must be the uint8 tensor the manifest's image_idx values refer to
    (trainval tensor for the val manifest, test tensor for the test manifest).
    """
    def __init__(self, images, manifest_path):
        with open(manifest_path) as f:
            m = json.load(f)
        self.meta = m["meta"]
        self.entries = m["entries"]
        self.images = images

    def __len__(self):
        return len(self.entries)

    def get_entry(self, i):
        return self.entries[i]

    def __getitem__(self, i):
        e = self.entries[i]
        clean = self.images[e["image_idx"]].float() / 255.0
        corrupted = apply_corruption(clean, e["params"])
        return corrupted, clean, e["label"], e["severity_level"], e["id"]
