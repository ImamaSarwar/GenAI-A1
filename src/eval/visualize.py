import matplotlib.pyplot as plt
import torch
from src.data.corruptions import CLASS_NAMES, LEVEL_NAMES


@torch.no_grad()
def show_recon(model, dataset, idxs, device="cuda", save_path=None):
    """Rows: clean target / corrupted input / reconstruction / absolute error map."""
    model.eval()
    fig, ax = plt.subplots(4, len(idxs), figsize=(2.1 * len(idxs), 8.6))
    for j, i in enumerate(idxs):
        x, y, lab, sev, _ = dataset[i]
        out = model(x.unsqueeze(0).to(device)).squeeze(0).cpu()
        err = (out - y).abs().mean(0)
        lvl = LEVEL_NAMES[sev] if sev >= 0 and lab != 0 else ""
        rows = [(y, "clean target"), (x, f"{CLASS_NAMES[lab]} {lvl}"), (out, "restored")]
        for r, (img, t) in enumerate(rows):
            ax[r, j].imshow(img.permute(1, 2, 0).clamp(0, 1).numpy()); ax[r, j].set_title(t, fontsize=8)
        ax[3, j].imshow(err.numpy(), cmap="inferno", vmin=0, vmax=0.3)
        ax[3, j].set_title(f"|error| mean={err.mean():.3f}", fontsize=8)
        for r in range(4): ax[r, j].axis("off")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
