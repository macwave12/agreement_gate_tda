"""Reproduce the figures and tables of "Agreement-Gated TDA: the cache that echoes CLIP, and a fix that adds
no parameters".

    python main.py --data /path/to/data_root

1. Features. CLIP ViT-B/16 image and text features of every dataset are extracted once on the GPU
   (features/<dataset>/), with the order in which TDA's runner would stream each split (seeds 0-2).
2. Streams. TDA is run on the cached features, one sample at a time, keeping for every sample CLIP's
   logits z, b = z - negative cache, and the positive cache's votes c. TDA, the agreement gate and
   their parts are then exact re-evaluations of these three (the caches do not depend on the output).
   DMN-weak and ECALP (Table 2) are the benchmark's own classes, run on the same cached features.
3. Figures and tables. Fig. 2 (diagnosis, val splits of the design sets), Fig. 3 (where the gate helps,
   all 16 test sets), Figs. 4-5 (which flips are right, held-out sets), Tables 1-4 and the two appendix
   tables (tables.md) -> results/.
"""
import argparse
import json
import pickle
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from online_method.tda import TDA  # noqa: E402
from online_method.tda_gate import ALPHA, BETA, TDAGate  # noqa: E402
from utils.tools import set_random_seed  # noqa: E402

REPO = Path(__file__).resolve().parent
SEEDS = (0, 1, 2)
MAIN_SEEDS = tuple(range(10))  # Table 1: ten stream orders of each design test set
# dataset: (set id of data/datautils.py, tier, name in the figures)
SETS = {
    "dtd": ("DTD", "design", "DTD"), "flower102": ("Flower102", "design", "Flowers"),
    "pets": ("Pets", "design", "Pets"), "eurosat": ("eurosat", "design", "EuroSAT"),
    "aircraft": ("Aircraft", "design", "Aircraft"),
    "caltech101": ("Caltech101", "near", "Caltech101"), "food101": ("Food101", "near", "Food101"),
    "cars": ("StanfordCarsWDS", "near", "Cars"),
    "gtsrb": ("GTSRB", "far", "GTSRB"), "imagenet_r": ("R", "far", "ImageNet-R"),
    "imagenet_sketch": ("K", "far", "ImageNet-Sketch"), "fer2013": ("FER2013", "far", "FER2013"),
    "mnist": ("MNIST", "far", "MNIST"), "pcam": ("PCAM", "far", "PCam"),
    "rendered_sst2": ("RenderedSST2", "far", "Rendered SST2"), "svhn": ("SVHN", "far", "SVHN"),
}
NO_FLIPS = ("imagenet_sketch",)  # 50,889 x 1000 classes: left out of the flip analysis (Figs. 4-5) and of Table 4


def splits_of(key):
    return ("val", "test") if SETS[key][1] == "design" else ("test",)


# --- 1. features --------------------------------------------------------------------------------
def class_names(set_id):
    wds = json.loads((REPO / "data" / "wds_classnames.json").read_text(encoding="utf-8"))
    wds_key = {"StanfordCarsWDS": "cars", "PCAM": "pcam", "RenderedSST2": "rendered_sst2"}.get(set_id, set_id.lower())
    if wds_key in wds:
        return wds[wds_key]
    if set_id in ("R", "K"):
        from data.imagenet_variants import imagenet_r_mask
        from data.imagnet_prompts import imagenet_classes
        return [c for c, m in zip(imagenet_classes, imagenet_r_mask) if m] if set_id == "R" else imagenet_classes
    from data import cls_to_names
    return getattr(cls_to_names, f"{set_id.lower()}_classes")


def _rgb(image):
    return image.convert("RGB")


def extract(key, data_root, out, gpu=0, workers=4):
    """Features, labels and stream orders of one dataset, built as online_tta.py builds its data and model."""
    import torchvision.transforms as T
    from clip.custom_clip import get_coop
    from data.datautils import build_dataset
    transform = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop((224, 224)), _rgb,
                           T.ToTensor(), T.Normalize([0.48145466, 0.4578275, 0.40821073],
                                                     [0.26862954, 0.26130258, 0.27577711])])
    set_id = SETS[key][0]
    data = {s: build_dataset(set_id, transform, data_root, mode=s) for s in splits_of(key)}
    out.mkdir(parents=True, exist_ok=True)
    model = None
    for seed in SEEDS:
        # the runner seeds, builds the data and the model (CLIP's init draws random numbers), then shuffles
        set_random_seed(seed)
        build_dataset(set_id, transform, data_root, mode="test")
        built = get_coop("ViT-B/16", class_names(set_id), gpu, 4, "a_photo_of_a").cuda(gpu).eval()
        state = torch.get_rng_state()
        for split, d in data.items():
            torch.set_rng_state(state)
            order = torch.cat([b.reshape(-1) for b in DataLoader(range(len(d)), batch_size=1, shuffle=True)])
            torch.save(order, out / f"perm_{split}_seed{seed}.pt")
        model = built if model is None else model
    with torch.no_grad():
        for split, d in data.items():
            feats, labels = [], []
            for images, target in DataLoader(d, batch_size=1, shuffle=False, num_workers=workers):
                images = images.cuda(gpu)
                with torch.cuda.amp.autocast():
                    f = model.image_encoder(images.type(model.dtype))
                    f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f.cpu())
                labels.append(target.reshape(-1))
            torch.save(torch.cat(feats), out / f"{split}_feats.pt")
            torch.save({"labels": torch.cat(labels).long()}, out / f"labels_{split}.pt")
            print(f"[{key}] {split}: {len(feats)} images", flush=True)
        with torch.cuda.amp.autocast():
            _, text, scale = model.forward_features(images)
        plain = model.get_text_features()  # outside autocast, as DMN-weak reads them
    torch.save({"text_features": text.cpu(), "logit_scale": torch.as_tensor(scale).cpu(),
                "text_features_plain": plain.cpu()}, out / "text_feats.pt")


# --- 2. streams ---------------------------------------------------------------------------------
class CachedFeatures:
    """Stands in for the CLIP model: `forward_features(i)` returns sample i's cached feature."""

    def __init__(self, feats, text):
        self.feats, self.text, self.scale = feats, text["text_features"], torch.as_tensor(text["logit_scale"])
        self.plain = text.get("text_features_plain")

    def get_text_features(self):
        return self.plain.clone()

    def eval(self):
        return self

    def forward_features(self, index):
        return self.feats[index:index + 1].clone(), self.text.clone(), self.scale.clone()


def stream_order(folder, split, seed):
    """The runner's own order for seeds 0-2 (saved with the features), a fixed CPU permutation for the others."""
    load = lambda name: torch.load(folder / name, map_location="cpu", weights_only=False)  # noqa: E731
    labels = load(f"labels_{split}.pt")["labels"].long()
    order = load(f"perm_{split}_seed{seed}.pt").long() if seed in SEEDS else \
        torch.randperm(len(labels), generator=torch.Generator().manual_seed(10_000 + seed))
    return order, labels[order]


def stream(folder, split, seed, read_first=False, alpha=ALPHA, beta=BETA):
    """Run TDA over one stream; returns float64 (z, b, c), the self-vote flags, the labels in stream order and
    the probabilities of TDA's own output (which it accumulates in fp16)."""
    load = lambda name: torch.load(folder / name, map_location="cpu", weights_only=False)  # noqa: E731
    order, labels = stream_order(folder, split, seed)
    tta = TDAGate(CachedFeatures(load(f"{split}_feats.pt"), load("text_feats.pt")), None, gate=False,
                  read_first=read_first, keep_trace=True, alpha=alpha, beta=beta)
    tta.prepare_model_and_optimization(None)
    out = [tta.adaptation_process(None, index, None)["output"] for index in order.tolist()]
    z, b, c, self_vote = zip(*tta.trace)
    return (torch.cat(z).double(), torch.cat(b).double(), torch.cat(c).double(), torch.tensor(self_vote), labels,
            torch.cat(out).float().softmax(-1).double())


def baseline(folder, name, seed, gpu=0):
    """Accuracy and ECE (%) of DMN-weak or ECALP on one test stream: the benchmark's classes, unchanged, on the
    cached features (on the GPU, under autocast, as the benchmark runs them)."""
    from online_method.dmn_weak import DMN_WEAK
    from online_method.ecalp import ECALP
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    load = lambda name: torch.load(folder / name, map_location=device, weights_only=False)  # noqa: E731
    order, labels = stream_order(folder, "test", seed)
    model = CachedFeatures(load("test_feats.pt"), load("text_feats.pt"))
    tta = {"dmn_weak": DMN_WEAK, "ecalp": ECALP}[name](model, device)
    tta.prepare_model_and_optimization(None)
    out = torch.cat([tta.adaptation_process(None, index, None)["output"].float().cpu() for index in order.tolist()])
    if name == "dmn_weak":  # the sum of two distributions
        out = out / out.sum(-1, keepdim=True)
    return score(out.double(), labels)[:2]


def ece(probs, labels, n_bins=20):
    """Expected calibration error (%), equal-width right-closed bins."""
    conf, correct = probs.max(-1).values.numpy(), (probs.argmax(-1) == labels).double().numpy()
    index = np.clip(np.digitize(conf, np.linspace(0.0, 1.0, n_bins + 1)[1:-1], right=True), 0, n_bins - 1)
    return 100 * sum((index == b).mean() * abs(correct[index == b].mean() - conf[index == b].mean())
                     for b in range(n_bins) if (index == b).any())


def score(probs, labels):
    """Accuracy (%), ECE (%) and mean confidence - accuracy (points)."""
    right = (probs.argmax(-1) == labels).double()
    return 100 * float(right.mean()), ece(probs, labels), 100 * float((probs.max(-1).values - right).mean())


def entropy(logits):
    lp = logits.log_softmax(-1)
    return -(lp.exp() * lp).sum(-1)


def zaec(adapted, zero_shot, iters=60):
    """ZAEC (Jiang et al., 2026): per sample, divide the adapted logits by the smallest T >= 1 that brings their
    entropy back up to zero-shot CLIP's (bisection); samples already at least as uncertain are left alone."""
    target = entropy(zero_shot)
    need = entropy(adapted) < target
    lo = torch.ones(len(adapted), dtype=adapted.dtype)
    hi = torch.ones_like(lo)
    for _ in range(64):  # grow hi until it brackets the target
        short = need & (entropy(adapted / hi[:, None]) < target)
        if not short.any():
            break
        hi = torch.where(short, hi * 2, hi)
    for _ in range(iters):
        mid = (lo + hi) / 2
        low = entropy(adapted / mid[:, None]) < target
        lo, hi = torch.where(need & low, mid, lo), torch.where(need & ~low, mid, hi)
    return adapted / torch.where(need, hi, torch.ones_like(hi))[:, None]


def fit_temperature(val_runs):
    """One temperature for TDA, fitted on the design sets' val streams by NLL (equal weight per dataset)."""
    def loss(t):
        per = {}
        for key, logits, labels in val_runs:
            per.setdefault(key, []).append(float(-(logits / t).log_softmax(-1).gather(1, labels[:, None]).mean()))
        return float(np.mean([np.mean(v) for v in per.values()]))
    best = min(np.exp(np.linspace(np.log(0.5), np.log(4), 61)), key=loss)
    return float(min(np.exp(np.linspace(np.log(best) - 0.04, np.log(best) + 0.04, 21)), key=loss))


def acc(logits, labels, mask):
    return 100 * float((logits[mask].argmax(-1) == labels[mask]).double().mean())


def gap(logits, labels, mask):
    """Mean confidence - accuracy (points): positive when overconfident."""
    p = logits[mask].softmax(-1)
    return 100 * float((p.max(-1).values - (p.argmax(-1) == labels[mask]).double()).mean())


def _corr(a, b):
    a, b = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-12)


def flip_scores(z, b, c, y):
    """One row per flip (TDA's decision differs from CLIP's) that is right (TDA right) or wrong (CLIP right),
    with ten label-free scores of CLIP, of the cache and of their relation."""
    p = z.softmax(-1)
    top2 = p.topk(2, dim=-1)
    p0, p1 = top2.indices[:, 0], (b + ALPHA * c).argmax(-1)
    c = c.clamp_min(0)
    pc = c / c.sum(-1, keepdim=True).clamp_min(1e-300)
    ct = pc.argmax(-1)
    pcs = 0.99 * pc + 0.01 / c.shape[1]
    m = 0.5 * (p + pcs)
    js = 0.5 * ((p * (p.clamp_min(1e-300).log() - m.log())).sum(-1) + (pcs * (pcs.log() - m.log())).sum(-1)) / np.log(2)
    at = lambda t, i: t.gather(1, i[:, None]).squeeze(1)  # noqa: E731
    lead, adv = at(b, p0) - at(b, ct), at(c, ct) - at(c, p0)
    s = {"clip_conf": top2.values[:, 0], "clip_margin": top2.values[:, 0] - top2.values[:, 1],
         "clip_p_cache_class": at(p, ct), "cache_conf": pc.max(-1).values, "cache_mass": c.max(-1).values,
         "cache_support": at(pc, p0), "rho": _corr(c, z), "one_minus_js": 1 - js,
         "weight_gap": at(pc, p0) - pc.max(-1).values,
         "alpha_needed": torch.where(adv > 0, lead / adv.clamp_min(1e-12), torch.full_like(lead, float("inf")))}
    right, wrong = (p1 != p0) & (p1 == y), (p1 != p0) & (p0 == y)
    keep = right | wrong
    return pd.DataFrame({k: v[keep].numpy() for k, v in s.items()}).assign(right=right[keep].numpy())


def summarize(folder, key, split, seed):
    """Everything the figures and tables need from one stream."""
    z, b, c, self_vote, y, tda_run = stream(folder, split, seed)
    tda = b + ALPHA * c
    pc = c.clamp_min(0)
    agree = (pc.sum(-1) > 0) & (pc.argmax(-1) == z.softmax(-1).topk(2, dim=-1).indices[:, 0])
    # the gate as the method applies it: closed where the cache's top class is CLIP's prediction (argmax of z, which
    # differs from `agree`'s softmax top-1 only where CLIP's two highest logits tie)
    gate = torch.where(((pc.sum(-1) > 0) & (c.argmax(-1) != z.argmax(-1)))[:, None], tda, b)
    # a control for the gate: the positive cache on as many samples as the gate opens on, drawn at random
    random = torch.zeros(len(y), dtype=torch.bool)
    n_open = round(float((~agree).double().mean()) * len(y))
    random[torch.randperm(len(y), generator=torch.Generator().manual_seed(seed))[:n_open]] = True
    # the cache ablation (Table 4): b plus the positive cache on some of the samples
    parts = {"clip": z, "base": b, "agree_only": torch.where(agree[:, None], tda, b),
             "disagree_only": torch.where(agree[:, None], b, tda), "random": torch.where(random[:, None], tda, b),
             "tda": tda, "gate": gate}
    out = {"dataset": key, "tier": SETS[key][1], "split": split, "seed": seed,
           "agree_share": 100 * float(agree.double().mean()),
           "p_agree_given_self": 100 * float(agree[self_vote].double().mean()),
           "p_agree_given_no_self": 100 * float(agree[~self_vote].double().mean())}
    for name, logits in parts.items():
        out[f"{name}_acc"], out[f"{name}_ece"], out[f"{name}_gap"] = score(logits.softmax(-1), y)
    for part, mask in (("agree", agree), ("disagree", ~agree)):
        for name, logits in (("base", b), ("tda", tda)):
            out[f"{name}_acc_{part}"], out[f"{name}_gap_{part}"] = acc(logits, y, mask), gap(logits, y, mask)
    flips, kept = None, None
    if SETS[key][1] == "design":
        if split == "test":  # TDA as it accumulates its own output (Table 1, Appendix A, Fig. 3)
            out["tda_run_acc"], out["tda_run_ece"], _ = score(tda_run, y)
        if seed in SEEDS:
            kept = (key, tda, y)  # for the temperature baseline, fitted on val and applied to test
            # TDA reading its caches before writing the sample to them (Fig. 2b, Table 2)
            _, b1, c1, _, _, _ = stream(folder, split, seed, read_first=True)
            out["read_first_acc"], out["read_first_ece"], _ = score((b1 + ALPHA * c1).softmax(-1), y)
        if seed in SEEDS and split == "test":
            out["tda_zaec_acc"], out["tda_zaec_ece"], _ = score(zaec(tda, z).softmax(-1), y)
            out["gate_zaec_acc"], out["gate_zaec_ece"], _ = score(zaec(parts["disagree_only"], z).softmax(-1), y)
            # Appendix A: TDA's published per-dataset alpha and beta instead of the single setting
            published = TDA(None, None).pos_params
            alpha, beta = published["alpha"][SETS[key][0]], published["beta"][SETS[key][0]]
            _, b1, c1, _, _, run1 = stream(folder, split, seed, alpha=alpha, beta=beta)
            out["tda_p1_acc"], out["tda_p1_ece"], _ = score(run1, y)
            gate1 = torch.where((c1.argmax(-1) != z.argmax(-1))[:, None], b1 + alpha * c1, b1)
            out["gate_p1_acc"], out["gate_p1_ece"], _ = score(gate1.softmax(-1), y)
    elif key not in NO_FLIPS:
        flips = flip_scores(z, b, c, y).assign(dataset=key, tier=SETS[key][1], seed=seed)
    return out, flips, kept


# --- 3. figures ---------------------------------------------------------------------------------
C_CLIP, C_BASE, C_TDA, C_GATE, C_READ_FIRST = "#4c78a8", "#8d99ae", "#e45756", "#2a9d8f", "#f4a261"
PAPER_RC = {"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6,
            "xtick.labelsize": 6.5, "ytick.labelsize": 6.5}


def _legend_above(ax, ncol):
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol, fontsize="small", frameon=False,
              handlelength=1.0, columnspacing=0.8)


def fig_selfvote(t, path):
    """Fig. 2a-b: agreement with and without a self-vote; ECE of CLIP, TDA and read-first TDA."""
    x = np.arange(len(t))
    fig, ax = plt.subplots(1, 2, figsize=(3.6, 2.0))
    w = 0.38
    ax[0].bar(x - w / 2, t.p_agree_given_self, w, color=C_TDA, label="with a self-vote")
    ax[0].bar(x + w / 2, t.p_agree_given_no_self, w, color=C_BASE, label="without")
    ax[0].set_title("(a) cache agrees with CLIP", pad=16)
    _legend_above(ax[0], 2)
    w = 0.27
    for i, (k, color, label) in enumerate((("clip", C_CLIP, "CLIP"), ("tda", C_TDA, "TDA"),
                                           ("read_first", C_READ_FIRST, "read-first TDA"))):
        ax[1].bar(x + (i - 1) * w, t[f"{k}_ece"], w, color=color, label=label)
    ax[1].set_ylabel("ECE (%)")
    ax[1].set_title("(b) ECE", pad=16)
    _legend_above(ax[1], 3)
    ax[0].set_ylabel("% of samples")
    for a in ax:
        a.set_xticks(x, t.name, rotation=40, ha="right")
        a.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_agreement(t, path):
    """Fig. 2c-d: change from adding TDA's positive cache to b, on agreeing and on disagreeing samples."""
    x = np.arange(len(t))
    fig, ax = plt.subplots(1, 2, figsize=(3.6, 2.2))
    w = 0.38
    share = {"agree": t.agree_share.mean(), "disagree": 100 - t.agree_share.mean()}
    for j, (metric, title) in enumerate((("acc", "accuracy change (pts)"), ("gap", "conf. $-$ acc. change (pts)"))):
        for i, (part, color, label) in enumerate((("agree", C_TDA, "agreeing, gate closed"),
                                                  ("disagree", C_GATE, "disagreeing, gate open"))):
            ax[j].bar(x + (i - 0.5) * w, t[f"tda_{metric}_{part}"] - t[f"base_{metric}_{part}"], w, color=color,
                      label=f"{label} ({share[part]:.0f}%)")
        ax[j].axhline(0, color="black", lw=0.6)
        ax[j].set_title(f"({'cd'[j]}) " + title)
        ax[j].set_xticks(x, t.name, rotation=40, ha="right")
        ax[j].grid(axis="y", alpha=0.3)
    handles, names = ax[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=2, frameon=False, fontsize="small", handlelength=1.0)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


TIER_STYLE = {"design": ("#525252", "o", "design (test)", 4), "near": ("#bdbdbd", "s", "near", 3),
              "far": ("white", "^", "far", 2)}
WHERE_NAMED = {"pets": (-8, 6), "caltech101": (-22, -16), "food101": (-28, -9), "cars": (4, -6), "imagenet_r": (7, -3),
               "gtsrb": (4, 2)}  # the sets the discussion names, with label offsets (points)


def fig_where(t, path):
    """Fig. 3: the gate's ECE change against CLIP's own confidence - accuracy gap, one point per dataset."""
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for tier, (color, marker, label, z) in TIER_STYLE.items():
        sub = t[t.tier == tier]
        if len(sub):
            ax.scatter(sub.clip_gap, sub.d_ece, marker=marker, s=28, color=color, edgecolors="black", linewidths=0.6,
                       label=label, zorder=z)
    for _, r in t[t.dataset.isin(WHERE_NAMED)].iterrows():
        ax.annotate(r["name"], (r.clip_gap, r.d_ece), textcoords="offset points", xytext=WHERE_NAMED[r.dataset],
                    fontsize="x-small")
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(0, color="black", lw=0.6, ls=":")
    ax.set_xlabel("CLIP confidence - accuracy (pts; uses labels)")
    ax.set_ylabel("ECE(gate) - ECE(TDA) (pts)\nbelow 0: the gate is better")
    ax.set_title(f"Spearman rho = {spearmanr(t.clip_gap, t.d_ece).correlation:+.2f} over {len(t)} datasets")
    ax.legend(loc="upper right", fontsize="small")
    ax.margins(x=0.06, y=0.14)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# score -> (whose it is, label); every score is label-free
SCORES = {
    "clip_conf": ("CLIP", "confidence"), "clip_margin": ("CLIP", "top-1 − top-2 margin"),
    "clip_p_cache_class": ("CLIP", "prob. of the cache's class"), "cache_conf": ("cache", "confidence"),
    "cache_mass": ("cache", "mass of its top class"), "cache_support": ("cache", "share on CLIP's class"),
    "rho": ("relation", "correlation over classes"), "one_minus_js": ("relation", "1 − JS divergence"),
    "weight_gap": ("relation", "share gap, CLIP's − own"), "alpha_needed": ("relation", "α needed to flip"),
}
DISCUSSED = ("clip_conf", "cache_conf", "one_minus_js")
TIER_LINE = {"near": ("#8c8c8c", "s", "--"), "far": ("#252525", "^", "-")}


def flip_tables(flips):
    """Per score and tier: the share of right flips by decile of the score's percentile within its dataset and
    seed, and the Spearman correlation between a flip being right and the score (mean over seeds, then datasets)."""
    deciles, corrs = [], []
    for key in SCORES:
        q = flips.groupby(["tier", "dataset", "seed"])[key].rank(pct=True, method="average")
        dec = np.minimum((q * 10).apply(np.ceil).astype(int) - 1, 9).clip(lower=0)
        for (tier, d), g in flips.assign(decile=dec).groupby(["tier", "decile"]):
            deciles.append({"score": key, "tier": tier, "decile": d, "n": len(g), "right_pct": 100 * g.right.mean()})
        for (tier, ds, seed), g in flips.groupby(["tier", "dataset", "seed"]):
            x, y = rankdata(g[key].replace(np.inf, np.nan).fillna(np.inf)), g.right.to_numpy(float)
            corrs.append({"score": key, "tier": tier, "dataset": ds,
                          "rho": np.corrcoef(x, y)[0, 1] if y.std() > 0 and x.std() > 0 else np.nan})
    corrs = pd.DataFrame(corrs).groupby(["score", "tier", "dataset"]).rho.mean().reset_index()
    return pd.DataFrame(deciles), corrs.groupby(["score", "tier"]).rho.mean().reset_index()


def fig_deciles(deciles, corrs, keys, path, ncols):
    """Figs. 4-5: share of right flips along each score, on the held-out tiers."""
    nrows = -(-len(keys) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.3 * nrows), sharey=True, squeeze=False)
    for ax, key in zip(axes.flat, keys):
        for i, (tier, (color, marker, style)) in enumerate(TIER_LINE.items()):
            d = deciles[(deciles.score == key) & (deciles.tier == tier)]
            ax.plot(10 * d.decile + 5, d.right_pct, marker=marker, ls=style, color=color, ms=3.5, lw=1.4, label=tier)
            rho = corrs[(corrs.score == key) & (corrs.tier == tier)].rho.item()
            ax.text(0.97, 0.08 + 0.1 * (1 - i), f"{tier}: $\\rho$ = {rho:+.2f}", ha="right", fontsize="small",
                    color=color, transform=ax.transAxes)
        ax.set_ylim(20, 100)
        ax.set_xlim(0, 100)
        ax.grid(alpha=0.3)
        ax.set_title("{}: {}".format(*SCORES[key]))
        ax.set_xlabel("score percentile within its dataset", fontsize="small")
    for ax in axes[:, 0]:
        ax.set_ylabel("% of flips that are right")
    axes[0, 0].legend(loc="lower left", fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --- 4. tables ----------------------------------------------------------------------------------
def num(x, d=2, sign=False):
    x = 0.0 if round(x, d) == 0 else x
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def per_seed(t, col):
    """seed x dataset values of one column, with the mean over the datasets as a last column."""
    wide = t.pivot(index="seed", columns="dataset", values=col)[list(dict.fromkeys(t.dataset))]
    return wide.assign(mean=wide.mean(axis=1))


def markdown(header, rows):
    return "\n".join("| " + " | ".join(r) + " |" for r in [header, ["---"] * len(header), *rows]) + "\n"


COMPARED = (("CLIP", "clip"), ("TDA", "tda"), ("agreement gate", "disagree_only"), ("read-first TDA", "read_first"),
            ("TDA + temperature (fit on val)", "tda_temp"), ("TDA + ZAEC", "tda_zaec"), ("gate + ZAEC", "gate_zaec"),
            ("DMN-weak", "dmn_weak"), ("ECALP", "ecalp"))
ABLATION = (("CLIP", "clip"), ("+ negative cache (b)", "base"), ("b + positive cache on agreeing samples", "agree_only"),
            ("b + positive cache on disagreeing samples (gate)", "disagree_only"),
            ("b + positive cache on as many random samples", "random"), ("b + positive cache on all samples (TDA)", "tda"))


def tables(runs):
    """Tables 1-4 and the appendix tables, as markdown, from the per-stream results."""
    test = runs[runs.split == "test"]
    design, three = test[test.tier == "design"], test[test.seed.isin(SEEDS)]
    names = [SETS[k][2] for k in dict.fromkeys(design.dataset)]
    out = []
    if len(design):
        rows = []
        for metric, label in (("acc", "Acc."), ("ece", "ECE")):
            for i, (method, col) in enumerate((("CLIP", "clip"), ("TDA", "tda_run"), ("agreement gate", "gate"))):
                wide = per_seed(design, f"{col}_{metric}")
                rows.append([label if i == 0 else "", method,
                             *[num(m) if col == "clip" else f"{num(m)} ± {num(sd)}"  # CLIP ignores the stream order
                               for m, sd in zip(wide.mean(), wide.std(ddof=1))]])
        out.append(f"### Table 1. Design test sets, {design.seed.nunique()} stream orders (mean ± std)\n\n"
                   + markdown(["", "method", *names, "Mean"], rows))
        d3 = design[design.seed.isin(SEEDS)]
        ref_acc, ref_ece = per_seed(d3, "tda_acc")["mean"], per_seed(d3, "tda_ece")["mean"]
        rows, per_set = [], []
        for method, col in COMPARED:
            a, e = per_seed(d3, f"{col}_acc"), per_seed(d3, f"{col}_ece")
            versus = ["--"] * 3 if col == "tda" else [num((a["mean"] - ref_acc).mean(), sign=True),
                                                      num((e["mean"] - ref_ece).mean(), sign=True),
                                                      f"{int((e['mean'] < ref_ece).sum())}/{len(e)}"]
            rows.append([method, num(a["mean"].mean()), num(e["mean"].mean()), *versus])
            per_set.append([method, *[num(v) for v in e.mean().iloc[:-1]]])
        out.append("### Table 2. Design test sets, 3 stream orders, mean of the five sets; changes are against TDA\n\n"
                   + markdown(["method", "Acc.", "ECE", "ΔAcc.", "ΔECE", "ECE wins"], rows))
    held = three[three.tier != "design"].groupby(["tier", "dataset"], sort=False).mean(numeric_only=True).reset_index()
    if len(held):
        cols = [f"{m}_{k}" for k in ("acc", "ece", "gap") for m in ("clip", "tda", "gate")]
        cells = lambda r: [num(r[c], 1, True) if c.endswith("gap") else num(r[c]) for c in cols]  # noqa: E731
        rows = []
        for tier, sub in held.groupby("tier", sort=False):
            rows += [[tier if i == 0 else "", SETS[r.dataset][2], *cells(r)] for i, (_, r) in enumerate(sub.iterrows())]
            rows.append(["", f"*mean ({len(sub)})*", *cells(sub[cols].mean())])
        out.append("### Table 3. Held-out sets, 3 stream orders: accuracy, ECE and confidence − accuracy gap\n\n"
                   + markdown(["tier", "dataset", *[f"{k} {m}" for k in ("acc.", "ECE", "gap")
                                                    for m in ("CLIP", "TDA", "gate")]], rows))
    tiers = three[~three.dataset.isin(NO_FLIPS)].groupby("tier", sort=False).mean(numeric_only=True)
    out.append("### Table 4. Cache ablation, 3 stream orders: accuracy / ECE (far: without ImageNet-Sketch)\n\n" + markdown(
        ["accuracy / ECE", *["design (test)" if t == "design" else t for t in tiers.index]],
        [*[[label, *[f"{num(r[f'{col}_acc'], 1)} / {num(r[f'{col}_ece'], 1)}" for _, r in tiers.iterrows()]]
           for label, col in ABLATION],
         ["samples the gate opens on (disagreeing)", *[f"{num(100 - r.agree_share, 0)}%" for _, r in tiers.iterrows()]]]))
    if len(design):
        rows = [[method, *[num(d3[f"{col}_{metric}"].mean()) for metric in ("acc", "ece") for col in (p1, p2)]]
                for method, p1, p2 in (("CLIP", "clip", "clip"), ("TDA", "tda_p1", "tda_run"),
                                       ("agreement gate", "gate_p1", "gate"))]
        out.append("### Appendix A. TDA's per-dataset alpha and beta against the single setting (design test, 3 orders)\n\n"
                   + markdown(["method", "acc. per-dataset", "acc. single", "ECE per-dataset", "ECE single"], rows))
        out.append("### Appendix B. ECE (%) of every method of Table 2 on each design test set\n\n"
                   + markdown(["ECE (%)", *names], per_set))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", help="dataset root (needed only for datasets whose features are not cached yet)")
    ap.add_argument("--features", default="features", help="feature cache")
    ap.add_argument("--out", default="results")
    ap.add_argument("--datasets", nargs="+", default=list(SETS), choices=list(SETS))
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)  # torch.cuda.amp.autocast, as upstream TDA calls it
    warnings.filterwarnings("ignore", message="Sparse invariant checks")  # ECALP's sparse graph
    feats, out = Path(a.features), Path(a.out)
    (out / "runs").mkdir(parents=True, exist_ok=True)

    def cached(name, compute):
        done = out / "runs" / f"{name}.pkl"
        if not done.exists():
            done.write_bytes(pickle.dumps(compute()))
            print(f"{name}: done", flush=True)
        return pickle.loads(done.read_bytes())

    rows, flips, kept = [], [], {"val": [], "test": []}
    for key in a.datasets:
        if not (feats / key / "text_feats.pt").exists():
            if not a.data:
                ap.error(f"no cached features for {key}: pass --data")
            extract(key, a.data, feats / key, a.gpu)
        design = SETS[key][1] == "design"
        for split in splits_of(key):
            for seed in (MAIN_SEEDS if design and split == "test" else SEEDS):
                row, f, k = cached(f"{key}_{split}_seed{seed}", lambda: summarize(feats / key, key, split, seed))
                if design and split == "test" and seed in SEEDS:
                    for name in ("dmn_weak", "ecalp"):  # Table 2's other training-free methods
                        row[f"{name}_acc"], row[f"{name}_ece"] = cached(
                            f"{key}_{name}_seed{seed}", lambda: baseline(feats / key, name, seed, a.gpu))
                rows.append(row)
                flips.append(f)
                if k is not None:
                    kept[split].append((len(rows) - 1, k))
    if kept["val"]:  # Table 2: TDA with one temperature, fitted on the val streams
        temperature = fit_temperature([k for _, k in kept["val"]])
        for i, (_, logits, labels) in kept["test"]:
            rows[i]["tda_temp_acc"], rows[i]["tda_temp_ece"], _ = score((logits / temperature).softmax(-1), labels)
    runs = pd.DataFrame(rows)
    # Fig. 3 measures the gate against TDA's own output where there is one (the design sets)
    runs["d_ece"] = runs.gate_ece - (runs.tda_run_ece.fillna(runs.tda_ece) if "tda_run_ece" in runs else runs.tda_ece)
    (out / "tables.md").write_text(tables(runs), encoding="utf-8")
    runs = runs[runs.seed.isin(SEEDS)]
    mean = runs.drop(columns="seed").groupby(["dataset", "tier", "split"], sort=False).mean().reset_index()
    mean["name"] = mean.dataset.map(lambda k: SETS[k][2])
    val, test = mean[mean.split == "val"], mean[mean.split == "test"].dropna(axis=1, how="all")
    with plt.rc_context(PAPER_RC):
        if len(val):
            val.to_csv(out / "fig2_data.csv", index=False)
            fig_selfvote(val, out / "paper_selfvote_val.png")
            fig_agreement(val, out / "paper_agreement_val.png")
        test.to_csv(out / "fig3_data.csv", index=False)
        fig_where(test, out / "paper_where_all.png")
        flips = [f for f in flips if f is not None]
        if {"near", "far"} <= {f.tier.iloc[0] for f in flips}:
            deciles, corrs = flip_tables(pd.concat(flips, ignore_index=True))
            deciles.to_csv(out / "fig45_deciles.csv", index=False)
            corrs.to_csv(out / "fig45_corr.csv", index=False)
            fig_deciles(deciles, corrs, DISCUSSED, out / "no_info_break_even.png", ncols=3)
            fig_deciles(deciles, corrs, list(SCORES), out / "no_info_scores.png", ncols=5)
    print("figures and tables in", out)


if __name__ == "__main__":
    main()
