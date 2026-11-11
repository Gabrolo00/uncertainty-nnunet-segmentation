"""
Autoencoder 3D sulla mappa di incertezza: impara l'incertezza attesa; il residuo (eccesso e mancanza rispetto
alla ricostruzione) stima quella inattesa. Training per gruppo di cross-fitting e inferenza sui casi non visti.

Uso: AE_DATASET=904 python autoencoder_train_infer.py train|infer|infer_resid --variant a --channels entropy --test_fold 0
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from autoencoder_config import DATASET_DIR, RUNS_DIR   # AE_DATASET sceglie il dataset (default 904)
D = DATASET_DIR
OUT = RUNS_DIR
PATCH = (64, 64, 64)
NORM = {"entropy": (0.0, 1.0), "mi": (0.0, 1.0), "logit": (0.0, 20.0), "dist": (0.0, 30.0), "ct": (-200.0, 500.0)}


# ---------------------------------------------------------------- dati
def load_case(c, channels, mmap=True):
    arrs = []
    for ch in channels:
        a = np.load(f"{D}/{c}/{ch}.npy", mmap_mode="r" if mmap else None)
        arrs.append(a)
    return arrs


def norm(x, ch):
    off, sc = NORM[ch]
    return (x.astype(np.float32) - off) / sc


class PatchSampler:
    """Patch 64^3: 75% centrate vicino al rene predetto (|dist| <= 10 mm), 25% ovunque nel ritaglio."""

    def __init__(self, cases, channels, seed):
        self.cases, self.channels = cases, channels
        self.rng = np.random.default_rng(seed)
        self.data = {c: load_case(c, channels) for c in cases}
        self.near = {}
        for c in cases:
            d = np.load(f"{D}/{c}/dist.npy", mmap_mode="r")
            idx = np.argwhere(np.abs(np.asarray(d[::2, ::2, ::2], dtype=np.float32)) <= 10) * 2
            self.near[c] = idx if len(idx) else None

    def patch(self):
        c = self.cases[self.rng.integers(len(self.cases))]
        arrs = self.data[c]
        shp = arrs[0].shape
        if self.near[c] is not None and self.rng.random() < 0.75:
            ctr = self.near[c][self.rng.integers(len(self.near[c]))]
        else:
            ctr = np.array([self.rng.integers(s) for s in shp])
        st = [int(np.clip(ct - p // 2, 0, max(s - p, 0))) for ct, p, s in zip(ctr, PATCH, shp)]
        out = np.zeros((len(arrs), *PATCH), np.float32)
        for i, (a, ch) in enumerate(zip(arrs, self.channels)):
            sub = norm(np.asarray(a[st[0]:st[0] + PATCH[0], st[1]:st[1] + PATCH[1], st[2]:st[2] + PATCH[2]]), ch)
            out[i, :sub.shape[0], :sub.shape[1], :sub.shape[2]] = sub
        for ax in range(3):  # flip casuali
            if self.rng.random() < 0.5:
                out = np.flip(out, axis=ax + 1)
        return np.ascontiguousarray(out)

    def batch(self, n):
        return torch.from_numpy(np.stack([self.patch() for _ in range(n)]))


# ---------------------------------------------------------------- modello
def block(ci, co, stride):
    return nn.Sequential(nn.Conv3d(ci, co, 3, stride, 1), nn.InstanceNorm3d(co, affine=True), nn.LeakyReLU(0.01, True),
                         nn.Conv3d(co, co, 3, 1, 1), nn.InstanceNorm3d(co, affine=True), nn.LeakyReLU(0.01, True))


def up(ci, co):
    return nn.Sequential(nn.ConvTranspose3d(ci, co, 2, 2), nn.InstanceNorm3d(co, affine=True), nn.LeakyReLU(0.01, True),
                         nn.Conv3d(co, co, 3, 1, 1), nn.InstanceNorm3d(co, affine=True), nn.LeakyReLU(0.01, True))


class AE(nn.Module):
    """n_down livelli di downsampling: 4 -> collo 4x4x4, 3 -> 8x8x8, 2 -> 16x16x16 (patch 64^3)."""

    def __init__(self, n_ch, latent, n_down=4):
        super().__init__()
        feats = (16, 32, 64, 128)[:n_down]
        enc, ci = [], n_ch
        for f in feats:
            enc.append(block(ci, f, 2))
            ci = f
        self.enc = nn.Sequential(*enc)
        self.to_lat = nn.Conv3d(ci, latent, 1)         # collo di bottiglia: latent canali su 4x4x4
        self.from_lat = nn.Conv3d(latent, ci, 1)
        dec = []
        for f in reversed(feats[:-1]):
            dec.append(up(ci, f))
            ci = f
        dec.append(up(ci, feats[0]))
        self.dec = nn.Sequential(*dec)
        self.head = nn.Conv3d(feats[0], n_ch, 1)

    def forward(self, x):
        z = self.to_lat(self.enc(x))
        return self.head(self.dec(self.from_lat(z)))


def wloss(rec, x):
    w = torch.ones_like(x)
    w[:, 0] = 1 + 9 * (x[:, 0] > 0.05).float()        # canale U: peso alla banda di incertezza
    return (w * (rec - x) ** 2).mean()


# ---------------------------------------------------------------- comandi
def run_dir(a):
    tag = f"{a.variant}_d{a.n_down}_lat{a.latent}_it{a.iters}"
    if a.train_frac < 1:   # curva di apprendimento: cartella separata, non sovrascrive la run completa
        tag += f"_frac{a.train_frac:g}"
    if a.train_list:   # AE solo su casi buoni ("good") o controllo casuale di pari dimensione ("rand")
        tag += "_good" if a.train_mode == "list" else "_rand"
    if a.inner_fold is not None:   # selezione degli iperparametri: gruppo di test mai toccato
        return f"{OUT}/selection_testfold{a.test_fold}/{tag}/inner{a.inner_fold}"
    return f"{OUT}/{tag}/fold{a.test_fold}"


def split(a):
    """Senza --inner_fold: training sui 4 gruppi != test_fold, applicazione a test_fold.
    Con --inner_fold k: training sui 3 gruppi diversi da test_fold e k, applicazione/valutazione al gruppo k
    (selezione annidata degli iperparametri: il gruppo test_fold non viene mai usato)."""
    f = pd.read_csv(f"{D}/folds.csv")
    excl = {a.test_fold} | ({a.inner_fold} if a.inner_fold is not None else set())
    tr = f[~f.fold.isin(excl)].case.tolist()
    te = f[f.fold == (a.inner_fold if a.inner_fold is not None else a.test_fold)].case.tolist()
    rng = np.random.default_rng(a.seed)
    rng.shuffle(tr)
    if a.train_frac < 1:  # curva di apprendimento
        tr = tr[: int(round(len(tr) * a.train_frac))]
    if a.train_list:  # filtro SOLO sui gruppi di training; il gruppo di test resta intero
        good = set(pd.read_csv(a.train_list).case)
        n_good = sum(c in good for c in tr)
        tr = [c for c in tr if c in good] if a.train_mode == "list" else tr[:n_good]   # tr e' gia' mescolato
        print(f"[split] train_mode={a.train_mode}: {len(tr)} casi di training (buoni nei gruppi: {n_good})", flush=True)
    nval = max(int(round(0.1 * len(tr))), 2)
    return tr[nval:], tr[:nval], te


def train(a):
    tr, va, te = split(a)
    od = run_dir(a)
    os.makedirs(od, exist_ok=True)
    chans = a.channels.split(",")
    dev = torch.device("cuda")
    torch.manual_seed(a.seed)
    net = AE(len(chans), a.latent, a.n_down).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.iters, pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda")
    t0 = time.time()
    s_tr = PatchSampler(tr, chans, a.seed)
    s_va = PatchSampler(va, chans, a.seed + 1)
    val_batches = [s_va.batch(a.batch) for _ in range(8)]          # validazione fissa
    print(f"[train] {a.variant} canali={chans} latent={a.latent} train={len(tr)} val={len(va)} test={len(te)} "
          f"(caricamento {time.time() - t0:.0f} s)", flush=True)
    log, best, t0 = [], np.inf, time.time()
    for it in range(1, a.iters + 1):
        net.train()
        x = s_tr.batch(a.batch).to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = wloss(net(x), x)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if it % a.val_every == 0 or it == a.iters:
            net.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                vl = float(np.mean([wloss(net(v.to(dev)), v.to(dev)).item() for v in val_batches]))
            log.append(dict(it=it, train_loss=float(loss.item()), val_loss=vl, minutes=(time.time() - t0) / 60))
            print(f"  it {it} train {loss.item():.5f} val {vl:.5f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
            if vl < best:
                best = vl
                torch.save(net.state_dict(), f"{od}/best.pt")
    pd.DataFrame(log).to_csv(f"{od}/train_log.csv", index=False)
    json.dump(dict(vars(a), train_cases=tr, val_cases=va, test_cases=te, best_val=best,
                   gpu_peak_GB=torch.cuda.max_memory_allocated(dev) / 1e9, minutes=(time.time() - t0) / 60),
              open(f"{od}/config.json", "w"), indent=1)
    print(f"[train] fine: best val {best:.5f}, {(time.time() - t0) / 60:.1f} min, "
          f"GPU {torch.cuda.max_memory_allocated(dev) / 1e9:.1f} GB", flush=True)


def gaussian(shape, sigma=1 / 8):
    g = [np.exp(-((np.arange(s) - (s - 1) / 2) ** 2) / (2 * (s * sigma) ** 2)) for s in shape]
    w = g[0][:, None, None] * g[1][None, :, None] * g[2][None, None, :]
    return (w / w.max()).astype(np.float32) + 1e-3


@torch.no_grad()
def infer(a):
    _, _, te = split(a)
    od = run_dir(a)
    chans = a.channels.split(",")
    dev = torch.device("cuda")
    net = AE(len(chans), a.latent, a.n_down).to(dev)
    net.load_state_dict(torch.load(f"{od}/best.pt", map_location=dev))
    net.eval()
    gw = gaussian(PATCH)
    step = [p // 2 for p in PATCH]
    t0 = time.time()
    for c in te:
        x = np.stack([norm(np.load(f"{D}/{c}/{ch}.npy"), ch) for ch in chans])
        shp = x.shape[1:]
        pad = [(0, max(p - s, 0)) for p, s in zip(PATCH, shp)]
        xp = np.pad(x, [(0, 0)] + pad)
        sp = xp.shape[1:]
        acc = np.zeros(sp, np.float32)
        wsum = np.zeros(sp, np.float32)
        starts = [sorted(set(list(range(0, s - p + 1, st)) + [s - p])) for s, p, st in zip(sp, PATCH, step)]
        for z in starts[0]:
            for y in starts[1]:
                for xx in starts[2]:
                    t = torch.from_numpy(xp[None, :, z:z + PATCH[0], y:y + PATCH[1], xx:xx + PATCH[2]]).to(dev)
                    with torch.autocast("cuda", dtype=torch.float16):
                        r = net(t)[0, 0].float().cpu().numpy()
                    acc[z:z + PATCH[0], y:y + PATCH[1], xx:xx + PATCH[2]] += r * gw
                    wsum[z:z + PATCH[0], y:y + PATCH[1], xx:xx + PATCH[2]] += gw
        rec = (acc / wsum)[: shp[0], : shp[1], : shp[2]]
        u = x[0]
        rec_u = rec * NORM[chans[0]][1] + NORM[chans[0]][0]
        u_raw = u * NORM[chans[0]][1] + NORM[chans[0]][0]
        cd = f"{od}/maps/{c}"
        os.makedirs(cd, exist_ok=True)
        np.save(f"{cd}/recon.npy", rec_u.astype(np.float16))
        np.save(f"{cd}/excess.npy", np.maximum(u_raw - rec_u, 0).astype(np.float16))
        np.save(f"{cd}/missing.npy", np.maximum(rec_u - u_raw, 0).astype(np.float16))
    print(f"[infer] {len(te)} casi, {(time.time() - t0) / 60:.1f} min", flush=True)


def sinus_fn(c, pr, gt):
    """FN dentro l'inviluppo convesso 3D dei reni predetti (>= 5 mL): proxy della convenzione del seno renale
    Usa la GT: SOLO per la valutazione. Cache in dataset/<caso>/sinus_fn.npy."""
    p = f"{D}/{c}/sinus_fn.npy"
    if os.path.exists(p):
        return np.load(p).astype(bool)
    from scipy.ndimage import label
    from skimage.morphology import convex_hull_image
    hull = np.zeros_like(pr)
    lab, n = label(pr)
    for k in range(1, n + 1):
        comp = lab == k
        if comp.sum() * 1.5 ** 3 / 1000 < 5:
            continue
        idx = np.argwhere(comp)
        bb = tuple(slice(i0, i1 + 1) for i0, i1 in zip(idx.min(0), idx.max(0)))
        hull[bb] |= convex_hull_image(comp[bb])
    s = hull & gt & ~pr
    np.save(p, s.astype(np.uint8))
    return s


@torch.no_grad()
def infer_resid(a):
    """Residuo |x - rec| sui canali DIVERSI da U (logit, ct), in unita' normalizzate, con lo stesso sliding window
    di infer(). Comando separato per non rifare l'inferenza delle mappe U gia' calcolate. Salva
    maps/<caso>/resid_<canale>.npy (fp16)."""
    _, _, te = split(a)
    od = run_dir(a)
    chans = a.channels.split(",")
    if len(chans) < 2:
        print("[infer_resid] solo il canale U: niente da fare")
        return
    dev = torch.device("cuda")
    net = AE(len(chans), a.latent, a.n_down).to(dev)
    net.load_state_dict(torch.load(f"{od}/best.pt", map_location=dev))
    net.eval()
    gw = gaussian(PATCH)
    step = [p // 2 for p in PATCH]
    for c in te:
        x = np.stack([norm(np.load(f"{D}/{c}/{ch}.npy"), ch) for ch in chans])
        shp = x.shape[1:]
        xp = np.pad(x, [(0, 0)] + [(0, max(p - s, 0)) for p, s in zip(PATCH, shp)])
        sp = xp.shape[1:]
        acc = np.zeros((len(chans) - 1, *sp), np.float32)
        wsum = np.zeros(sp, np.float32)
        starts = [sorted(set(list(range(0, s - p + 1, st)) + [s - p])) for s, p, st in zip(sp, PATCH, step)]
        for z in starts[0]:
            for y in starts[1]:
                for xx in starts[2]:
                    t = torch.from_numpy(xp[None, :, z:z + PATCH[0], y:y + PATCH[1], xx:xx + PATCH[2]]).to(dev)
                    with torch.autocast("cuda", dtype=torch.float16):
                        r = net(t)[0, 1:].float().cpu().numpy()
                    acc[:, z:z + PATCH[0], y:y + PATCH[1], xx:xx + PATCH[2]] += r * gw
                    wsum[z:z + PATCH[0], y:y + PATCH[1], xx:xx + PATCH[2]] += gw
        rec = (acc / wsum)[:, : shp[0], : shp[1], : shp[2]]
        for i, ch in enumerate(chans[1:]):
            np.save(f"{od}/maps/{c}/resid_{ch}.npy", np.abs(x[i + 1] - rec[i]).astype(np.float16))
    print(f"[infer_resid] {len(te)} casi, canali {chans[1:]}", flush=True)


def quickeval(a):
    """Valutazione rapida (pilota e selezione): AUROC e AUPRC voxel-wise errori contro corretti, dentro il
    ritaglio, ROI 15 mm, lontano dal bordo (|dist| > w). Gli FN del seno renale (convenzione) sono ESCLUSI dalla
    valutazione (ne' positivi ne' negativi). Non sostituisce la valutazione completa (autoencoder_eval.py)."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    from scipy.ndimage import distance_transform_edt
    _, _, te = split(a)
    od = run_dir(a)
    f = pd.read_csv(f"{D}/folds.csv").set_index("case")
    rows = []
    for c in te:
        pr, gt = np.load(f"{D}/{c}/pred.npy").astype(bool), np.load(f"{D}/{c}/gt.npy").astype(bool)
        dist = np.load(f"{D}/{c}/dist.npy").astype(np.float32)
        roi = distance_transform_edt(~(pr | gt), sampling=1.5) <= 15
        roi &= ~sinus_fn(c, pr, gt)
        err = (pr ^ gt) & roi
        maps = {"entropy": np.load(f"{D}/{c}/entropy.npy"), "mi": np.load(f"{D}/{c}/mi.npy"),
                "excess": np.load(f"{od}/maps/{c}/excess.npy"), "missing": np.load(f"{od}/maps/{c}/missing.npy")}
        maps["excess+missing"] = maps["excess"].astype(np.float32) + maps["missing"]
        maps["recon_err_abs"] = np.abs(np.load(f"{D}/{c}/entropy.npy").astype(np.float32)
                                       - np.load(f"{od}/maps/{c}/recon.npy").astype(np.float32))
        for w in [2.0, 4.5]:
            m = roi & (np.abs(dist) > w)
            y = err[m]
            if y.sum() < 10 or (~y).sum() < 10:
                continue
            for k, v in maps.items():
                s = v[m].astype(np.float32)
                rows.append(dict(case=c, source=f.loc[c, "source"], w=w, map=k, n_err=int(y.sum()),
                                 auroc=roc_auc_score(y, s), auprc=average_precision_score(y, s)))
    d = pd.DataFrame(rows)
    d.to_csv(f"{od}/quickeval_per_case.csv", index=False)
    s = d.groupby(["source", "w", "map"])[["auroc", "auprc"]].median().round(4)
    s.to_csv(f"{od}/quickeval_summary.csv")
    print(s.to_string())
    print("casi valutati:", d.groupby(["source", "w"]).case.nunique().to_dict())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["train", "infer", "quickeval", "infer_resid"])
    ap.add_argument("--variant", required=True)
    ap.add_argument("--channels", required=True, help="il primo e' la mappa U, es. entropy,logit,ct")
    ap.add_argument("--test_fold", type=int, required=True)
    ap.add_argument("--latent", type=int, default=16)
    ap.add_argument("--n_down", type=int, default=4, help="livelli di downsampling: 4 -> collo 4^3, 3 -> 8^3")
    ap.add_argument("--inner_fold", type=int, default=None, help="selezione annidata degli iperparametri")
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_every", type=int, default=250)
    ap.add_argument("--train_frac", type=float, default=1.0, help="frazione dei casi di training (curva di apprendimento)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_list", default="", help="CSV con colonna case: casi ammessi nel training")
    ap.add_argument("--train_mode", choices=["list", "random_like_list"], default="list",
                    help="list = solo i casi della lista; random_like_list = controllo, casi casuali di pari numero")
    a = ap.parse_args()
    {"train": train, "infer": infer, "quickeval": quickeval, "infer_resid": infer_resid}[a.cmd](a)


if __name__ == "__main__":
    main()
