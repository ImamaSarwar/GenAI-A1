"""
Runtime corruptions for Tasks 1-3.

Every corruption is described by a small JSON-serialisable `params` dict, so the
exact same corruption can be re-applied later (this is what the manifests store).

Images: float32 torch tensors (3, H, W) in [0, 1].
"""
import math
import numpy as np
import torch
import torchvision.transforms.functional as TF

IMG_SIZE = 128

# class labels (order matches [p_clean, p_salt, p_blur, p_occlusion])
CLEAN, SALT, BLUR, OCC = 0, 1, 2, 3
CLASS_NAMES = ["clean", "salt_pepper", "blur", "occlusion"]
LEVEL_NAMES = ["low", "medium", "high"]

# fixed test severities from the assignment
TEST_LEVELS = {
    SALT: [0.03, 0.08, 0.15],                       # probability
    BLUR: [(3, 0.7), (5, 1.5), (7, 2.5)],           # (kernel, sigma)
    OCC:  [(0.10, 1), (0.20, 2), (0.35, 3)],        # (area fraction, n rectangles)
}


# ------------------------------------------------------------------ occlusion
def _overlaps(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def sample_rects(total_frac, n_rects, rng, size=IMG_SIZE, max_tries=200):
    """
    Sample n non-overlapping rectangles [x, y, w, h] that JOINTLY cover ~total_frac
    of the image. The area is split equally between rectangles; each rectangle gets
    a random aspect ratio in [0.5, 2] and a random position.
    Non-overlap is enforced by rejection sampling so the covered area is accurate
    (if no placement is found after max_tries, overlap is accepted as a fallback).
    """
    area_each = total_frac * size * size / n_rects
    rects = []
    for _ in range(n_rects):
        placed = None
        for _try in range(max_tries):
            ar = math.exp(rng.uniform(math.log(0.5), math.log(2.0)))
            w = int(round(math.sqrt(area_each * ar)))
            w = min(max(w, 1), size)
            h = int(round(area_each / w))
            h = min(max(h, 1), size)
            x = int(rng.integers(0, size - w + 1))
            y = int(rng.integers(0, size - h + 1))
            cand = [x, y, w, h]
            if all(not _overlaps(cand, r) for r in rects):
                placed = cand
                break
        rects.append(placed if placed is not None else cand)
    return rects


def rects_coverage(rects, size=IMG_SIZE):
    """Actual fraction of the image covered by the union of the rectangles."""
    mask = np.zeros((size, size), dtype=bool)
    for x, y, w, h in rects:
        mask[y:y + h, x:x + w] = True
    return float(mask.mean())


# ------------------------------------------------------------- apply functions
def apply_salt_pepper(img, p, seed):
    """Select pixels with prob p; each becomes white or black with prob 0.5."""
    rng = np.random.default_rng(seed)
    _, H, W = img.shape
    selected = rng.random((H, W)) < p
    white = rng.random((H, W)) < 0.5
    out = img.clone()
    out[:, torch.from_numpy(selected & white)] = 1.0
    out[:, torch.from_numpy(selected & ~white)] = 0.0
    return out


def apply_blur(img, kernel, sigma):
    return TF.gaussian_blur(img, kernel_size=[int(kernel), int(kernel)],
                            sigma=[float(sigma), float(sigma)])


def apply_occlusion(img, rects):
    out = img.clone()
    for x, y, w, h in rects:
        out[:, y:y + h, x:x + w] = 0.0   # black rectangle
    return out


def apply_corruption(img, params):
    """Apply a corruption described by `params` to a clean image tensor."""
    t = params["type"]
    if t == "clean":
        return img.clone()
    if t == "salt_pepper":
        return apply_salt_pepper(img, params["p"], params["seed"])
    if t == "blur":
        return apply_blur(img, params["kernel"], params["sigma"])
    if t == "occlusion":
        return apply_occlusion(img, params["rects"])
    raise ValueError(f"unknown corruption type: {t}")


# ------------------------------------------------------------ parameter sampling
def sample_params(label, rng, size=IMG_SIZE):
    """Random (training / validation) parameters, using the assignment's ranges."""
    seed = int(rng.integers(0, 2**31 - 1))
    if label == CLEAN:
        return {"type": "clean", "seed": seed}
    if label == SALT:
        return {"type": "salt_pepper", "p": float(rng.uniform(0.02, 0.15)), "seed": seed}
    if label == BLUR:
        return {"type": "blur", "kernel": int(rng.choice([3, 5, 7])),
                "sigma": float(rng.uniform(0.5, 2.5)), "seed": seed}
    if label == OCC:
        n = int(rng.integers(1, 4))                    # 1, 2 or 3 rectangles
        frac = float(rng.uniform(0.10, 0.35))          # joint area fraction
        rects = sample_rects(frac, n, rng, size)
        return {"type": "occlusion", "n_rects": n, "target_frac": frac,
                "coverage": rects_coverage(rects, size), "rects": rects, "seed": seed}
    raise ValueError(label)


def fixed_params(label, level, rng, size=IMG_SIZE):
    """Fixed-severity parameters for the final test set (level 0/1/2)."""
    seed = int(rng.integers(0, 2**31 - 1))
    if label == SALT:
        return {"type": "salt_pepper", "p": TEST_LEVELS[SALT][level], "seed": seed}
    if label == BLUR:
        k, s = TEST_LEVELS[BLUR][level]
        return {"type": "blur", "kernel": k, "sigma": s, "seed": seed}
    if label == OCC:
        frac, n = TEST_LEVELS[OCC][level]
        rects = sample_rects(frac, n, rng, size)
        return {"type": "occlusion", "n_rects": n, "target_frac": frac,
                "coverage": rects_coverage(rects, size), "rects": rects, "seed": seed}
    raise ValueError(label)
