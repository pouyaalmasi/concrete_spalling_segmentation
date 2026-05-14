# train_unet_spalling_meta_final.py
# ============================================================
# FINAL end-to-end script:
# 1) Build patch pools + patch-type stats
# 2) PSO meta-search:
#       - curriculum (sampling over patch types)
#       - loss mix alpha (BCE vs Dice)
#       - with recall constraint (inspection-style)
# 3) Full training with best policy
# 4) Automatic threshold selection on VAL (sweep)
# 5) TEST evaluation (global pixel-wise) + plots
# 6) Save overlays + predicted masks (VAL + TEST samples)
# 7) Save all logs, configs, curves, and extra paper-friendly figures
# ============================================================

import os, glob, random, math, json, time
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# Config (edit these only)
# ============================================================
ROOT = r"C:\Users\arina\Desktop\Spalling Paper\Final_Dataset"  # expects train/val/test/images+mask folders

RESULTS_DIR = "results_meta_unet"
PATCH_SIZE = 512

# Pool building
TRAIN_CANDIDATES_PER_IMAGE = 35
VAL_CANDIDATES_PER_IMAGE = 12
MIN_POS_FRAC_KEEP = 0.001  # drop ultra-weak positives in pool building (keeps pool cleaner)

# Training patch sampler
TRAIN_SAMPLES_PER_EPOCH = 8000
VAL_SAMPLES_PER_EPOCH = 1200

# Meta-search (PSO)
PSO_PARTICLES = 10
PSO_ITERS = 10
BURST_STEPS = 120
BURST_BATCH = 6
BURST_LR = 8e-4
POS_WEIGHT_VALUE = 5.0

# Constraint + objective shaping
RECALL_CONSTRAINT = 0.9
OVERFIT_PENALTY = 0.15

# Full training
EPOCHS = 35
BATCH_SIZE = 6
LR = 1e-3

# Evaluation and saving visuals
N_VIS_OVERLAYS_VAL = 30
N_VIS_OVERLAYS_TEST = 50

# If your full images are large, set TILE_INFERENCE=True
TILE_INFERENCE = False
TILE_SIZE = 512
TILE_OVERLAP = 64

SEED = 42


# ============================================================
# Utils
# ============================================================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def softmax_np(x, eps=1e-12):
    x = np.array(x, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + eps)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def overlay_mask_on_image(rgb, mask01, alpha=0.45):
    """
    Consistent overlay:
      - True mask pixels highlighted in GREEN-ish (by adding to G channel)
      - Predicted mask pixels highlighted in RED-ish (by adding to R channel)
    Here we only overlay one mask (e.g., prediction).
    We'll create red overlay for predicted; green overlay for GT in combined visuals.
    """
    out = rgb.copy().astype(np.float32)
    m = (mask01 > 0.5).astype(np.float32)

    # Red overlay for mask
    overlay = out.copy()
    overlay[..., 0] = np.clip(overlay[..., 0] + 255.0 * m, 0, 255)

    out = (1 - alpha) * out + alpha * overlay
    return out.astype(np.uint8)


def overlay_gt_pred(rgb, gt01, pred01, alpha=0.45):
    """
    Show both GT and Pred:
      - GT in GREEN channel
      - Pred in RED channel
      - Overlap becomes yellow-ish
    """
    out = rgb.copy().astype(np.float32)
    gt = (gt01 > 0.5).astype(np.float32)
    pr = (pred01 > 0.5).astype(np.float32)

    overlay = out.copy()
    overlay[..., 1] = np.clip(overlay[..., 1] + 255.0 * gt, 0, 255)  # green for GT
    overlay[..., 0] = np.clip(overlay[..., 0] + 255.0 * pr, 0, 255)  # red for Pred

    out = (1 - alpha) * out + alpha * overlay
    return out.astype(np.uint8)


def confusion_from_logits(logits, targets, thr=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()
    y = targets.float()

    tp = (preds * y).sum().item()
    tn = ((1 - preds) * (1 - y)).sum().item()
    fp = (preds * (1 - y)).sum().item()
    fn = ((1 - preds) * y).sum().item()
    return tp, tn, fp, fn


def metrics_from_confusion(tp, tn, fp, fn, eps=1e-9):
    iou = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = (2 * prec * rec) / (prec + rec + eps)
    return {
        "IoU": float(iou),
        "Dice": float(dice),
        "PixelAcc": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1": float(f1),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


@torch.no_grad()
def evaluate_loader_global(model, loader, device, alpha_unused=None, thr=0.5):
    model.eval()
    tp = tn = fp = fn = 0.0
    loss_sum = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)

        _tp, _tn, _fp, _fn = confusion_from_logits(logits, y, thr=thr)
        tp += _tp; tn += _tn; fp += _fp; fn += _fn
        n_batches += 1

    m = metrics_from_confusion(tp, tn, fp, fn)
    return m


def plot_training_curves(history, save_path):
    # Loss, IoU, PixelAcc, Recall
    epochs = history["epoch"]

    fig = plt.figure(figsize=(18, 4))
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Val")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.grid(True, alpha=0.3); ax1.legend()

    ax2 = fig.add_subplot(1, 4, 2)
    ax2.plot(epochs, history["train_iou"], label="Train")
    ax2.plot(epochs, history["val_iou"], label="Val")
    ax2.set_title("IoU"); ax2.set_xlabel("Epoch"); ax2.grid(True, alpha=0.3); ax2.legend()

    ax3 = fig.add_subplot(1, 4, 3)
    ax3.plot(epochs, history["train_acc"], label="Train")
    ax3.plot(epochs, history["val_acc"], label="Val")
    ax3.set_title("Pixel Accuracy"); ax3.set_xlabel("Epoch"); ax3.grid(True, alpha=0.3); ax3.legend()

    ax4 = fig.add_subplot(1, 4, 4)
    ax4.plot(epochs, history["train_rec"], label="Train")
    ax4.plot(epochs, history["val_rec"], label="Val")
    ax4.set_title("Recall"); ax4.set_xlabel("Epoch"); ax4.grid(True, alpha=0.3); ax4.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_pso_convergence(pso_log_df, save_path):
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(pso_log_df["iter"], pso_log_df["gbest_fitness"])
    ax.set_title("PSO Convergence (lower is better)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Global Best Fitness")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_sampling_probs(best_sampling, save_path):
    labels = list(best_sampling.keys())
    values = [best_sampling[k] for k in labels]

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(labels, values)
    ax.set_title("Learned Curriculum: Patch-Type Sampling Probabilities")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, max(0.01, 1.05 * max(values)))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(tp, tn, fp, fn, save_path):
    cm = np.array([[tn, fp],
                   [fn, tp]], dtype=np.float64)

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix (pixels)")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["GT 0", "GT 1"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(cm[i,j])}", ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


@torch.no_grad()
def threshold_sweep(model, loader, device, thr_list, recall_constraint=0.0):
    """
    Pick threshold that maximizes IoU under optional recall constraint.
    Returns best_thr, a dataframe with all results.
    """
    rows = []
    best_thr = thr_list[0]
    best_iou = -1.0

    for thr in thr_list:
        m = evaluate_loader_global(model, loader, device, thr=thr)
        rows.append({
            "thr": thr,
            "IoU": m["IoU"],
            "Dice": m["Dice"],
            "PixelAcc": m["PixelAcc"],
            "Precision": m["Precision"],
            "Recall": m["Recall"],
            "F1": m["F1"],
        })
        if m["Recall"] >= recall_constraint and m["IoU"] > best_iou:
            best_iou = m["IoU"]
            best_thr = thr

    df = pd.DataFrame(rows)
    return best_thr, df


def plot_threshold_sweep(df, best_thr, save_path):
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df["thr"], df["IoU"], label="IoU")
    ax.plot(df["thr"], df["Recall"], label="Recall")
    ax.plot(df["thr"], df["Precision"], label="Precision")
    ax.axvline(best_thr, linestyle="--", label=f"Best thr={best_thr:.2f}")
    ax.set_title("Threshold Sweep on Validation")
    ax.set_xlabel("Threshold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# ============================================================
# Patch typing for curriculum
# ============================================================
EDGE_POS     = 0
TINY_POS     = 1
NEG_TEXTURE  = 2
NEG_SHADOW   = 3
OTHER        = 4

TYPE_NAMES = {
    EDGE_POS: "EDGE_POS",
    TINY_POS: "TINY_POS",
    NEG_TEXTURE: "NEG_TEXTURE",
    NEG_SHADOW: "NEG_SHADOW",
    OTHER: "OTHER",
}

def mask_edge_strength(mask01):
    m = (mask01 > 0.5).astype(np.uint8) * 255
    if m.max() == 0:
        return 0.0
    k = np.ones((3, 3), np.uint8)
    dil = cv2.dilate(m, k, iterations=1)
    ero = cv2.erode(m, k, iterations=1)
    grad = cv2.absdiff(dil, ero)
    return float((grad > 0).mean())


def patch_texture_score(rgb_patch):
    g = cv2.cvtColor(rgb_patch, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    return float(lap.var())


def patch_shadow_score(rgb_patch):
    g = cv2.cvtColor(rgb_patch, cv2.COLOR_RGB2GRAY)
    return float(g.mean())


def classify_patch_type(img_patch, mask_patch01, pos_frac, edge_frac,
                        tiny_pos_thresh=0.002, edge_thresh=0.01,
                        texture_thresh=120.0, shadow_thresh=85.0):
    if pos_frac > 0:
        if pos_frac <= tiny_pos_thresh:
            return TINY_POS
        if edge_frac >= edge_thresh:
            return EDGE_POS
        return OTHER

    tex = patch_texture_score(img_patch)
    sh  = patch_shadow_score(img_patch)
    if tex >= texture_thresh:
        return NEG_TEXTURE
    if sh <= shadow_thresh:
        return NEG_SHADOW
    return OTHER


def read_full(img_path, msk_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 0).astype(np.float32)
    return img, mask


def random_crop_coords(H, W, ps):
    y0 = random.randint(0, H - ps)
    x0 = random.randint(0, W - ps)
    return x0, y0


def build_patch_pool(image_paths, mask_paths,
                     patch_size=512,
                     candidates_per_image=40,
                     min_pos_frac=0.001,
                     seed=42):
    seed_all(seed)
    pool = []

    for i in tqdm(range(len(image_paths)), desc="Building patch pool"):
        img, mask = read_full(image_paths[i], mask_paths[i])
        H, W = mask.shape
        ps = patch_size

        if H < ps or W < ps:
            img = cv2.resize(img, (max(W, ps), max(H, ps)), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (max(W, ps), max(H, ps)), interpolation=cv2.INTER_NEAREST)
            H, W = mask.shape

        for _ in range(candidates_per_image):
            x0, y0 = random_crop_coords(H, W, ps)
            img_p = img[y0:y0+ps, x0:x0+ps]
            msk_p = mask[y0:y0+ps, x0:x0+ps]
            pos_frac = float(msk_p.mean())
            edge_frac = mask_edge_strength(msk_p)

            t = classify_patch_type(img_p, msk_p, pos_frac, edge_frac)

            if pos_frac > 0 and pos_frac < min_pos_frac:
                continue

            pool.append((i, x0, y0, t, pos_frac))

    return pool


def patch_pool_stats(pool):
    counts = {}
    pos_fracs = []
    for (_, _, _, t, pf) in pool:
        counts[t] = counts.get(t, 0) + 1
        pos_fracs.append(pf)
    total = len(pool)
    stats = {
        "total_patches": total,
        "type_counts": {TYPE_NAMES[k]: int(v) for k, v in counts.items()},
        "pos_frac_mean": float(np.mean(pos_fracs)) if len(pos_fracs) else 0.0,
        "pos_frac_median": float(np.median(pos_fracs)) if len(pos_fracs) else 0.0,
        "pos_frac_p95": float(np.percentile(pos_fracs, 95)) if len(pos_fracs) else 0.0,
    }
    return stats


def plot_patch_pool_type_counts(stats, save_path):
    type_counts = stats["type_counts"]
    labels = list(type_counts.keys())
    values = [type_counts[k] for k in labels]

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(labels, values)
    ax.set_title("Patch Pool Composition (counts by type)")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# ============================================================
# Dataset that draws from patch pool using sampling weights
# ============================================================
class SpallingPatchPoolDataset(Dataset):
    def __init__(self, image_paths, mask_paths, patch_pool,
                 patch_size=512, augment=None,
                 sampling_probs=None,
                 samples_per_epoch=8000):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.pool = patch_pool
        self.ps = patch_size
        self.augment = augment
        self.samples_per_epoch = samples_per_epoch

        self.by_type = {}
        for j, (_, _, _, t, _) in enumerate(self.pool):
            self.by_type.setdefault(t, []).append(j)

        self.types_present = sorted(list(self.by_type.keys()))
        self.set_sampling_probs(sampling_probs)

    def set_sampling_probs(self, sampling_probs):
        if sampling_probs is None:
            p = {t: 1.0 / len(self.types_present) for t in self.types_present}
        else:
            s = 0.0
            p = {}
            for t in self.types_present:
                p[t] = float(sampling_probs.get(t, 0.0))
                s += p[t]
            if s <= 0:
                p = {t: 1.0 / len(self.types_present) for t in self.types_present}
            else:
                p = {t: p[t] / s for t in self.types_present}
        self.sampling_probs = p

        self._type_list = list(self.types_present)
        self._type_prob = np.array([self.sampling_probs[t] for t in self._type_list], dtype=np.float64)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        t = np.random.choice(self._type_list, p=self._type_prob)
        pool_idx = random.choice(self.by_type[t])
        img_i, x0, y0, _, _ = self.pool[pool_idx]

        img, mask = read_full(self.image_paths[img_i], self.mask_paths[img_i])
        H, W = mask.shape
        ps = self.ps
        if H < ps or W < ps:
            img = cv2.resize(img, (max(W, ps), max(H, ps)), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (max(W, ps), max(H, ps)), interpolation=cv2.INTER_NEAREST)

        img_p = img[y0:y0+ps, x0:x0+ps]
        msk_p = mask[y0:y0+ps, x0:x0+ps]

        if self.augment is not None:
            aug = self.augment(image=img_p, mask=msk_p)
            img_p, msk_p = aug["image"], aug["mask"]

        img_p = img_p.astype(np.float32) / 255.0
        img_p = np.transpose(img_p, (2, 0, 1))
        msk_p = np.expand_dims(msk_p.astype(np.float32), axis=0)

        return torch.tensor(img_p), torch.tensor(msk_p)


# ============================================================
# U-Net
# ============================================================
def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=(64, 128, 256, 512)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        ch = in_channels
        for f in features:
            self.downs.append(conv_block(ch, f))
            ch = f

        self.bottleneck = conv_block(features[-1], features[-1] * 2)

        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.ups.append(conv_block(f * 2, f))

        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i // 2]
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)

        return self.final(x)


# ============================================================
# Loss: alpha*BCE + (1-alpha)*Dice
# ============================================================
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum(dim=(2, 3))
        den = (probs + targets).sum(dim=(2, 3)) + self.eps
        return (1 - (num / den)).mean()

def combined_loss(logits, targets, bce_loss_fn, dice_loss_fn, alpha):
    return alpha * bce_loss_fn(logits, targets) + (1 - alpha) * dice_loss_fn(logits, targets)


# ============================================================
# Train / Val loops (log confusion-derived metrics)
# ============================================================
def train_one_epoch(model, loader, optim, bce, dice, device, alpha, thr=0.5):
    model.train()
    loss_sum = 0.0
    tp = tn = fp = fn = 0.0
    n = 0

    for x, y in tqdm(loader, leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optim.zero_grad(set_to_none=True)

        logits = model(x)
        loss = combined_loss(logits, y, bce, dice, alpha)
        loss.backward()
        optim.step()

        loss_sum += loss.item()
        _tp, _tn, _fp, _fn = confusion_from_logits(logits.detach(), y, thr=thr)
        tp += _tp; tn += _tn; fp += _fp; fn += _fn
        n += 1

    m = metrics_from_confusion(tp, tn, fp, fn)
    m["Loss"] = float(loss_sum / max(n, 1))
    return m

@torch.no_grad()
def validate(model, loader, bce, dice, device, alpha, thr=0.5):
    model.eval()
    loss_sum = 0.0
    tp = tn = fp = fn = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = combined_loss(logits, y, bce, dice, alpha)

        loss_sum += loss.item()
        _tp, _tn, _fp, _fn = confusion_from_logits(logits, y, thr=thr)
        tp += _tp; tn += _tn; fp += _fp; fn += _fn
        n += 1

    m = metrics_from_confusion(tp, tn, fp, fn)
    m["Loss"] = float(loss_sum / max(n, 1))
    return m


# ============================================================
# Simple PSO
# ============================================================
class PSO:
    def __init__(self, dim, bounds, n_particles=10, iters=8,
                 w=0.6, c1=1.4, c2=1.4, seed=42):
        self.dim = dim
        self.bounds = bounds
        self.np = n_particles
        self.iters = iters
        self.w = w
        self.c1 = c1
        self.c2 = c2
        rng = np.random.RandomState(seed)

        self.x = np.zeros((self.np, dim), dtype=np.float64)
        self.v = np.zeros((self.np, dim), dtype=np.float64)

        for i in range(self.np):
            for d in range(dim):
                lo, hi = bounds[d]
                self.x[i, d] = rng.uniform(lo, hi)

        self.pbest = self.x.copy()
        self.pbest_val = np.full((self.np,), np.inf, dtype=np.float64)
        self.gbest = self.x[0].copy()
        self.gbest_val = np.inf
        self.rng = rng

        self.log = []

    def step(self, fitness_fn, it_index):
        for i in range(self.np):
            val = fitness_fn(self.x[i])
            if val < self.pbest_val[i]:
                self.pbest_val[i] = val
                self.pbest[i] = self.x[i].copy()
            if val < self.gbest_val:
                self.gbest_val = val
                self.gbest = self.x[i].copy()

        self.log.append({"iter": it_index, "gbest_fitness": float(self.gbest_val)})

        for i in range(self.np):
            r1 = self.rng.rand(self.dim)
            r2 = self.rng.rand(self.dim)
            self.v[i] = (self.w * self.v[i] +
                         self.c1 * r1 * (self.pbest[i] - self.x[i]) +
                         self.c2 * r2 * (self.gbest - self.x[i]))
            self.x[i] = self.x[i] + self.v[i]

            for d in range(self.dim):
                lo, hi = self.bounds[d]
                self.x[i, d] = float(np.clip(self.x[i, d], lo, hi))

    def run(self, fitness_fn, verbose=True):
        for it in range(1, self.iters + 1):
            self.step(fitness_fn, it_index=it)
            if verbose:
                print(f"[PSO] iter {it}/{self.iters} best = {self.gbest_val:.6f}")
        return self.gbest, self.gbest_val, pd.DataFrame(self.log)


# ============================================================
# Meta-evaluation: short burst training
# ============================================================
def evaluate_candidate(
    candidate_vec,
    device,
    train_ds,
    val_loader,
    base_seed=123,
    burst_steps=120,
    burst_batch=6,
    lr=8e-4,
    pos_weight_value=5.0,
    recall_constraint=0.92,
    overfit_penalty=0.15,
    thr=0.5
):
    seed_all(base_seed)

    w_edge, w_tiny, w_tex, w_shadow, alpha = candidate_vec
    alpha = float(np.clip(alpha, 0.05, 0.95))

    probs4 = softmax_np([w_edge, w_tiny, w_tex, w_shadow])
    p_map = {EDGE_POS: probs4[0], TINY_POS: probs4[1], NEG_TEXTURE: probs4[2], NEG_SHADOW: probs4[3]}

    train_ds.set_sampling_probs(p_map)
    train_loader = DataLoader(train_ds, batch_size=burst_batch, shuffle=True, num_workers=0, pin_memory=True)

    model = UNet().to(device)
    pos_weight = torch.tensor([pos_weight_value], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    dice = DiceLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    step = 0
    tr_tp = tr_tn = tr_fp = tr_fn = 0.0

    for x, y in train_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optim.zero_grad(set_to_none=True)

        logits = model(x)
        loss = combined_loss(logits, y, bce, dice, alpha)
        loss.backward()
        optim.step()

        _tp, _tn, _fp, _fn = confusion_from_logits(logits.detach(), y, thr=thr)
        tr_tp += _tp; tr_tn += _tn; tr_fp += _fp; tr_fn += _fn

        step += 1
        if step >= burst_steps:
            break

    tr_m = metrics_from_confusion(tr_tp, tr_tn, tr_fp, tr_fn)
    va_m = evaluate_loader_global(model, val_loader, device, thr=thr)

    gap = max(0.0, tr_m["IoU"] - va_m["IoU"])

    if va_m["Recall"] < recall_constraint:
        penalty = (recall_constraint - va_m["Recall"]) * 5.0
    else:
        penalty = 0.0

    score = (va_m["IoU"] + 0.25 * va_m["Recall"]) - overfit_penalty * gap
    fitness = -(score) + penalty
    return float(fitness)


# ============================================================
# Full image inference (optional tiling)
# ============================================================
@torch.no_grad()
def predict_full_image(model, rgb, device, thr=0.5,
                       tile_inference=False, tile_size=512, overlap=64):
    model.eval()
    H, W, _ = rgb.shape

    if not tile_inference:
        x = rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]
        x = torch.from_numpy(x).to(device)
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        pred = (prob > thr).astype(np.uint8)
        return pred, prob

    # Tiled inference with overlap
    stride = tile_size - overlap
    pred_prob = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)

    for y0 in range(0, H, stride):
        for x0 in range(0, W, stride):
            y1 = min(y0 + tile_size, H)
            x1 = min(x0 + tile_size, W)

            y0a = max(0, y1 - tile_size)
            x0a = max(0, x1 - tile_size)
            patch = rgb[y0a:y1, x0a:x1]

            x = patch.astype(np.float32) / 255.0
            x = np.transpose(x, (2, 0, 1))[None, ...]
            x = torch.from_numpy(x).to(device)

            logits = model(x)
            prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

            pred_prob[y0a:y1, x0a:x1] += prob
            weight[y0a:y1, x0a:x1] += 1.0

    pred_prob = pred_prob / np.maximum(weight, 1e-6)
    pred = (pred_prob > thr).astype(np.uint8)
    return pred, pred_prob


def save_overlays_for_split(model, image_paths, mask_paths, device,
                            out_dir, thr=0.5, max_items=50,
                            tile_inference=False, tile_size=512, overlap=64):
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "pred_masks"))
    ensure_dir(os.path.join(out_dir, "overlays"))

    n = min(len(image_paths), max_items)
    idxs = list(range(len(image_paths)))
    random.shuffle(idxs)
    idxs = idxs[:n]

    for k, i in enumerate(tqdm(idxs, desc=f"Saving overlays -> {out_dir}")):
        img_path = image_paths[i]
        msk_path = mask_paths[i]
        base = os.path.splitext(os.path.basename(img_path))[0]

        rgb, gt01 = read_full(img_path, msk_path)
        pred01, prob = predict_full_image(model, rgb, device, thr=thr,
                                          tile_inference=tile_inference,
                                          tile_size=tile_size,
                                          overlap=overlap)

        # Save predicted mask PNG (0/255)
        pred_png = (pred01 * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, "pred_masks", f"{base}_pred.png"), pred_png)

        # Save overlay (GT green, Pred red)
        overlay = overlay_gt_pred(rgb, gt01, pred01, alpha=0.45)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_dir, "overlays", f"{base}_overlay.png"), overlay_bgr)

print("torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))
    print("Current device:", torch.cuda.current_device())

# ============================================================
# Main
# ============================================================
def main():
    seed_all(SEED)
    ensure_dir(RESULTS_DIR)

    # Output folders
    dirs = {
        "plots": os.path.join(RESULTS_DIR, "plots"),
        "overlays_val": os.path.join(RESULTS_DIR, "overlays_val"),
        "overlays_test": os.path.join(RESULTS_DIR, "overlays_test"),
        "models": os.path.join(RESULTS_DIR, "models"),
        "logs": os.path.join(RESULTS_DIR, "logs"),
        "configs": os.path.join(RESULTS_DIR, "configs"),
    }
    for d in dirs.values():
        ensure_dir(d)

    # Load split paths
    train_imgs = sorted(glob.glob(os.path.join(ROOT, "train", "images", "*")))
    train_msks = sorted(glob.glob(os.path.join(ROOT, "train", "masks", "*")))
    val_imgs   = sorted(glob.glob(os.path.join(ROOT, "val", "images", "*")))
    val_msks   = sorted(glob.glob(os.path.join(ROOT, "val", "masks", "*")))
    test_imgs  = sorted(glob.glob(os.path.join(ROOT, "test", "images", "*")))
    test_msks  = sorted(glob.glob(os.path.join(ROOT, "test", "masks", "*")))

    assert len(train_imgs) == len(train_msks), "Train image/mask mismatch"
    assert len(val_imgs) == len(val_msks), "Val image/mask mismatch"
    assert len(test_imgs) == len(test_msks), "Test image/mask mismatch"
    print(f"Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # -------------------------
    # Build patch pools
    # -------------------------
    print("\nBuilding TRAIN patch pool ...")
    train_pool = build_patch_pool(
        train_imgs, train_msks,
        patch_size=PATCH_SIZE,
        candidates_per_image=TRAIN_CANDIDATES_PER_IMAGE,
        min_pos_frac=MIN_POS_FRAC_KEEP,
        seed=SEED
    )
    train_pool_stats_dict = patch_pool_stats(train_pool)
    save_json(train_pool_stats_dict, os.path.join(dirs["configs"], "train_pool_stats.json"))
    plot_patch_pool_type_counts(train_pool_stats_dict, os.path.join(dirs["plots"], "train_pool_type_counts.png"))

    print("Building VAL patch pool ...")
    val_pool = build_patch_pool(
        val_imgs, val_msks,
        patch_size=PATCH_SIZE,
        candidates_per_image=VAL_CANDIDATES_PER_IMAGE,
        min_pos_frac=0.0,
        seed=777
    )
    val_pool_stats_dict = patch_pool_stats(val_pool)
    save_json(val_pool_stats_dict, os.path.join(dirs["configs"], "val_pool_stats.json"))
    plot_patch_pool_type_counts(val_pool_stats_dict, os.path.join(dirs["plots"], "val_pool_type_counts.png"))

    # Deterministic-ish validation dataset (uniform sampling over pool)
    val_ds = SpallingPatchPoolDataset(
        val_imgs, val_msks, val_pool,
        patch_size=PATCH_SIZE,
        augment=None,
        sampling_probs=None,
        samples_per_epoch=VAL_SAMPLES_PER_EPOCH
    )
    val_loader = DataLoader(val_ds, batch_size=BURST_BATCH, shuffle=False, num_workers=0, pin_memory=True)

    # Training augmentations
    train_aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=20, p=0.7),
        A.RandomBrightnessContrast(p=0.5),
    ])

    # Training dataset (pool-based sampler)
    train_ds = SpallingPatchPoolDataset(
        train_imgs, train_msks, train_pool,
        patch_size=PATCH_SIZE,
        augment=train_aug,
        sampling_probs=None,
        samples_per_epoch=TRAIN_SAMPLES_PER_EPOCH
    )

    # -------------------------
    # PSO meta-search
    # -------------------------
    bounds = [
        (0.1, 5.0),  # w_edge
        (0.1, 5.0),  # w_tiny
        (0.1, 5.0),  # w_texture
        (0.1, 5.0),  # w_shadow
        (0.05, 0.95) # alpha
    ]

    print("\n=== PSO META-SEARCH (curriculum + loss-mix) ===")
    pso = PSO(dim=5, bounds=bounds, n_particles=PSO_PARTICLES, iters=PSO_ITERS, seed=2026)

    def fitness_fn(vec):
        return evaluate_candidate(
            candidate_vec=vec,
            device=device,
            train_ds=train_ds,
            val_loader=val_loader,
            base_seed=123,
            burst_steps=BURST_STEPS,
            burst_batch=BURST_BATCH,
            lr=BURST_LR,
            pos_weight_value=POS_WEIGHT_VALUE,
            recall_constraint=RECALL_CONSTRAINT,
            overfit_penalty=OVERFIT_PENALTY,
            thr=0.5
        )

    best_vec, best_fit, pso_log_df = pso.run(fitness_fn, verbose=True)
    pso_log_df.to_csv(os.path.join(dirs["logs"], "pso_log.csv"), index=False)
    plot_pso_convergence(pso_log_df, os.path.join(dirs["plots"], "pso_convergence.png"))

    w_edge, w_tiny, w_tex, w_shadow, alpha = best_vec
    probs4 = softmax_np([w_edge, w_tiny, w_tex, w_shadow])
    best_sampling = {
        "EDGE_POS": float(probs4[0]),
        "TINY_POS": float(probs4[1]),
        "NEG_TEXTURE": float(probs4[2]),
        "NEG_SHADOW": float(probs4[3]),
    }
    alpha = float(alpha)

    print("\n=== BEST FOUND ===")
    print("alpha (BCE mix):", alpha)
    for k, v in best_sampling.items():
        print(f"  {k:<12}: {v:.3f}")

    plot_sampling_probs(best_sampling, os.path.join(dirs["plots"], "learned_sampling_probs.png"))

    # Apply best sampling policy
    train_ds.set_sampling_probs({
        EDGE_POS: best_sampling["EDGE_POS"],
        TINY_POS: best_sampling["TINY_POS"],
        NEG_TEXTURE: best_sampling["NEG_TEXTURE"],
        NEG_SHADOW: best_sampling["NEG_SHADOW"],
    })

    # Save meta config (paper reproducibility)
    meta_config = {
        "alpha": alpha,
        "sampling_probs": best_sampling,
        "pso_best_fitness": float(best_fit),
        "patch_size": PATCH_SIZE,
        "train_candidates_per_image": TRAIN_CANDIDATES_PER_IMAGE,
        "val_candidates_per_image": VAL_CANDIDATES_PER_IMAGE,
        "train_samples_per_epoch": TRAIN_SAMPLES_PER_EPOCH,
        "val_samples_per_epoch": VAL_SAMPLES_PER_EPOCH,
        "recall_constraint": RECALL_CONSTRAINT,
        "overfit_penalty": OVERFIT_PENALTY,
        "pos_weight_value": POS_WEIGHT_VALUE,
        "tile_inference": TILE_INFERENCE,
        "tile_size": TILE_SIZE,
        "tile_overlap": TILE_OVERLAP,
        "seed": SEED,
    }
    save_json(meta_config, os.path.join(dirs["configs"], "meta_config.json"))

    # -------------------------
    # Full training with best policy
    # -------------------------
    model = UNet().to(device)
    pos_weight = torch.tensor([POS_WEIGHT_VALUE], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    dice = DiceLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=LR)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    # keep val loader stable (don’t reuse BURST_BATCH, use BATCH_SIZE for speed)
    val_loader_full = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    history = {
        "epoch": [],
        "train_loss": [], "val_loss": [],
        "train_iou": [],  "val_iou": [],
        "train_acc": [],  "val_acc": [],
        "train_prec": [], "val_prec": [],
        "train_rec": [],  "val_rec": [],
        "train_f1": [],   "val_f1": [],
    }

    best_val_iou = -1.0
    best_path = os.path.join(dirs["models"], "unet_spalling_meta_best.pth")

    # Live plotting (optional)
    plt.ion()
    live_fig = plt.figure(figsize=(16, 4))

    def live_plot(history_dict):
        e = history_dict["epoch"]
        live_fig.clf()

        ax1 = live_fig.add_subplot(1, 4, 1)
        ax1.plot(e, history_dict["train_loss"], label="Train")
        ax1.plot(e, history_dict["val_loss"], label="Val")
        ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.grid(True, alpha=0.3); ax1.legend()

        ax2 = live_fig.add_subplot(1, 4, 2)
        ax2.plot(e, history_dict["train_iou"], label="Train")
        ax2.plot(e, history_dict["val_iou"], label="Val")
        ax2.set_title("IoU"); ax2.set_xlabel("Epoch"); ax2.grid(True, alpha=0.3); ax2.legend()

        ax3 = live_fig.add_subplot(1, 4, 3)
        ax3.plot(e, history_dict["train_acc"], label="Train")
        ax3.plot(e, history_dict["val_acc"], label="Val")
        ax3.set_title("PixelAcc"); ax3.set_xlabel("Epoch"); ax3.grid(True, alpha=0.3); ax3.legend()

        ax4 = live_fig.add_subplot(1, 4, 4)
        ax4.plot(e, history_dict["train_rec"], label="Train")
        ax4.plot(e, history_dict["val_rec"], label="Val")
        ax4.set_title("Recall"); ax4.set_xlabel("Epoch"); ax4.grid(True, alpha=0.3); ax4.legend()

        live_fig.tight_layout()
        live_fig.canvas.draw()
        live_fig.canvas.flush_events()
        plt.pause(0.001)

    print("\n=== FULL TRAINING ===")
    for epoch in range(1, EPOCHS + 1):
        tr_m = train_one_epoch(model, train_loader, optim, bce, dice, device, alpha, thr=0.5)
        va_m = validate(model, val_loader_full, bce, dice, device, alpha, thr=0.5)

        print(
            f"Epoch {epoch:02d} | "
            f"Train loss {tr_m['Loss']:.4f} IoU {tr_m['IoU']:.3f} Acc {tr_m['PixelAcc']:.4f} "
            f"Prec {tr_m['Precision']:.3f} Rec {tr_m['Recall']:.3f} | "
            f"Val loss {va_m['Loss']:.4f} IoU {va_m['IoU']:.3f} Acc {va_m['PixelAcc']:.4f} "
            f"Prec {va_m['Precision']:.3f} Rec {va_m['Recall']:.3f}"
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_m["Loss"]); history["val_loss"].append(va_m["Loss"])
        history["train_iou"].append(tr_m["IoU"]);   history["val_iou"].append(va_m["IoU"])
        history["train_acc"].append(tr_m["PixelAcc"]); history["val_acc"].append(va_m["PixelAcc"])
        history["train_prec"].append(tr_m["Precision"]); history["val_prec"].append(va_m["Precision"])
        history["train_rec"].append(tr_m["Recall"]); history["val_rec"].append(va_m["Recall"])
        history["train_f1"].append(tr_m["F1"]); history["val_f1"].append(va_m["F1"])

        # live plot + also keep saving a static curves image
        live_plot(history)
        plot_training_curves(history, os.path.join(dirs["plots"], "training_curves.png"))

        if va_m["IoU"] > best_val_iou:
            best_val_iou = va_m["IoU"]
            torch.save(model.state_dict(), best_path)
            print("  ✅ Saved best model:", best_path)

    # Save training logs
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(dirs["logs"], "training_log.csv"), index=False)

    plt.ioff()
    plt.close(live_fig)

    # -------------------------
    # Reload best model for evaluation
    # -------------------------
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    # -------------------------
    # Threshold sweep on VAL to pick best threshold
    # -------------------------
    # Create a VAL loader that is stable (same as val_loader_full already)
    thr_list = [round(x, 2) for x in np.linspace(0.2, 0.8, 31)]
    best_thr, thr_df = threshold_sweep(model, val_loader_full, device, thr_list, recall_constraint=RECALL_CONSTRAINT)
    thr_df.to_csv(os.path.join(dirs["logs"], "val_threshold_sweep.csv"), index=False)
    plot_threshold_sweep(thr_df, best_thr, os.path.join(dirs["plots"], "val_threshold_sweep.png"))
    save_json({"best_threshold": float(best_thr)}, os.path.join(dirs["configs"], "best_threshold.json"))
    print("\nBest threshold from VAL sweep:", best_thr)

    # -------------------------
    # TEST evaluation (patch-based global pixel-wise)
    #   We evaluate on:
    #     - patch-based test loader (fair w.r.t training regime)
    #     - and also full-image overlays/preds for qualitative reporting
    # -------------------------
    # Build a patch pool for TEST (for metrics on patch distribution)
    print("\nBuilding TEST patch pool for evaluation metrics ...")
    test_pool = build_patch_pool(
        test_imgs, test_msks,
        patch_size=PATCH_SIZE,
        candidates_per_image=VAL_CANDIDATES_PER_IMAGE,
        min_pos_frac=0.0,
        seed=999
    )
    test_pool_stats_dict = patch_pool_stats(test_pool)
    save_json(test_pool_stats_dict, os.path.join(dirs["configs"], "test_pool_stats.json"))
    plot_patch_pool_type_counts(test_pool_stats_dict, os.path.join(dirs["plots"], "test_pool_type_counts.png"))

    test_ds = SpallingPatchPoolDataset(
        test_imgs, test_msks, test_pool,
        patch_size=PATCH_SIZE,
        augment=None,
        sampling_probs=None,
        samples_per_epoch=VAL_SAMPLES_PER_EPOCH  # same scale as val for stability
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    test_metrics = evaluate_loader_global(model, test_loader, device, thr=best_thr)
    print("\n=== TEST SET (patch-based global pixel-wise) ===")
    for k in ["IoU", "Dice", "PixelAcc", "Precision", "Recall", "F1"]:
        print(f"{k:<10}: {test_metrics[k]:.4f}")
    print(f"TP/TN/FP/FN: {test_metrics['TP']}/{test_metrics['TN']}/{test_metrics['FP']}/{test_metrics['FN']}")

    save_json(test_metrics, os.path.join(dirs["logs"], "test_metrics.json"))
    plot_confusion_matrix(
        tp=test_metrics["TP"], tn=test_metrics["TN"], fp=test_metrics["FP"], fn=test_metrics["FN"],
        save_path=os.path.join(dirs["plots"], "test_confusion_matrix.png")
    )

    # Also record VAL metrics at best threshold (for paper tables)
    val_metrics_bestthr = evaluate_loader_global(model, val_loader_full, device, thr=best_thr)
    save_json(val_metrics_bestthr, os.path.join(dirs["logs"], "val_metrics_bestthr.json"))

    # -------------------------
    # Save qualitative overlays (VAL + TEST full images)
    # -------------------------
    print("\nSaving VAL overlays (full images) ...")
    save_overlays_for_split(
        model, val_imgs, val_msks, device,
        out_dir=dirs["overlays_val"],
        thr=best_thr,
        max_items=N_VIS_OVERLAYS_VAL,
        tile_inference=TILE_INFERENCE,
        tile_size=TILE_SIZE,
        overlap=TILE_OVERLAP
    )

    print("Saving TEST overlays (full images) ...")
    save_overlays_for_split(
        model, test_imgs, test_msks, device,
        out_dir=dirs["overlays_test"],
        thr=best_thr,
        max_items=N_VIS_OVERLAYS_TEST,
        tile_inference=TILE_INFERENCE,
        tile_size=TILE_SIZE,
        overlap=TILE_OVERLAP
    )

    # -------------------------
    # Final “run summary” json
    # -------------------------
    run_summary = {
        "device": device,
        "best_val_iou_during_training_thr0.5": float(best_val_iou),
        "best_threshold_from_val_sweep": float(best_thr),
        "val_metrics_at_best_thr": val_metrics_bestthr,
        "test_metrics_patch_based_at_best_thr": test_metrics,
        "outputs": {
            "best_model": best_path,
            "training_log_csv": os.path.join(dirs["logs"], "training_log.csv"),
            "pso_log_csv": os.path.join(dirs["logs"], "pso_log.csv"),
            "meta_config": os.path.join(dirs["configs"], "meta_config.json"),
            "plots_dir": dirs["plots"],
            "overlays_val_dir": dirs["overlays_val"],
            "overlays_test_dir": dirs["overlays_test"],
        }
    }
    save_json(run_summary, os.path.join(RESULTS_DIR, "run_summary.json"))

    print("\n✅ DONE. Key outputs saved under:", RESULTS_DIR)
    print(" - results_meta_unet/models/unet_spalling_meta_best.pth")
    print(" - results_meta_unet/logs/training_log.csv")
    print(" - results_meta_unet/logs/pso_log.csv")
    print(" - results_meta_unet/plots/training_curves.png")
    print(" - results_meta_unet/plots/pso_convergence.png")
    print(" - results_meta_unet/plots/learned_sampling_probs.png")
    print(" - results_meta_unet/plots/val_threshold_sweep.png")
    print(" - results_meta_unet/plots/test_confusion_matrix.png")
    print(" - results_meta_unet/overlays_val/overlays/*.png")
    print(" - results_meta_unet/overlays_test/overlays/*.png")


if __name__ == "__main__":
    main()
