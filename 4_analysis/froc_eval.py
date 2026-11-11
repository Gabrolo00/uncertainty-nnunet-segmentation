"""
Confronto delle mappe a parita' di volume di falsi allarmi: frazione di zone di errore lontane dal bordo
trovate al variare della soglia.

Uso: python froc_eval.py --unc_types entropy mutual_information DEU_entropy ebcu_entropy --out_dir <out> [--border_mm 2 4.5]
"""
import argparse
import os
import sys
from multiprocessing import Pool

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import label

sys.path[:0] = ['/home/grolo/thesis_pipeline/3_uncertainty/evaluation']  # cartelle dei moduli del progetto importati qui
from evaluate_threshold import normalize_minmax_full, compute_signed_distance_transform  # noqa: E402

S26 = np.ones((3, 3, 3), dtype=bool)
MASS = ["s0543", "s0545"]
FA_POINTS = {"zones": [0.5, 1.0], "volume": [1.0, 2.0, 5.0]}   # zone/paziente, ml/paziente


def process_case(job):
    case, measures, a = job
    pi = nib.load(os.path.join(a.pred_dir, f"{case}.nii.gz"))
    sp = np.array(pi.header.get_zooms()[:3], float)
    pred = np.asarray(pi.dataobj) > 0.5
    gt = np.asarray(nib.load(os.path.join(a.gt_dir, f"{case}.nii.gz")).dataobj) > 0.5
    union = pred | gt
    if not union.any():
        return [], []
    idx = np.argwhere(union)
    mrg = np.ceil(20.0 / sp).astype(int)
    sl = tuple(slice(max(lo - m, 0), min(hi + m + 1, s)) for lo, hi, m, s in zip(idx.min(0), idx.max(0), mrg, union.shape))
    p, g = pred[sl], gt[sl]
    roi = compute_signed_distance_transform(p | g, sp) >= -15.0
    ad = np.abs(compute_signed_distance_transform(p, sp)) if p.any() else np.full(p.shape, np.inf)
    err = (p ^ g) & roi
    nb = {w: roi & (ad > w) for w in a.widths}
    comps = {}
    for w in a.widths:
        lab, n = label(err & nb[w], structure=S26)
        sz = np.bincount(lab.ravel())
        comps[w] = (lab, [k for k in range(1, n + 1) if sz[k] >= a.min_vox])

    comp_rows, fa_rows = [], []
    for ut in measures:
        up = os.path.join(a.unc_dir, f"{case}_{ut}.nii.gz")
        if not os.path.exists(up):
            continue
        u = normalize_minmax_full(np.asarray(nib.load(up).dataobj, dtype=np.float32))[sl]
        for w in a.widths:
            lab, keep = comps[w]
            for k in keep:
                v = u[lab == k]
                comp_rows.append(dict(case=case, uncertainty_type=ut, w=w, comp=k, size=int(v.size),
                                      u_max=float(v.max()), u_p90=float(np.percentile(v, 90))))
            m = nb[w]
            um, em = u[m], err[m]   # solo i voxel della ROI no-border: le somme di volume sono veloci
            for t in a.grid:
                fz = nz = np.nan
                if a.with_zones:  # costoso (un labeling 3D per soglia), asse secondario
                    flag = (u >= t) & m
                    flab, nf = label(flag, structure=S26)
                    fsz = np.bincount(flab.ravel())
                    touch = np.zeros(nf + 1, bool)
                    touch[np.unique(flab[flag & err])] = True
                    big = fsz >= a.min_vox
                    big[0] = False
                    fz, nz = int((big & ~touch).sum()), int(big.sum())
                fa_rows.append(dict(case=case, uncertainty_type=ut, w=w, t=t,
                                    false_zones=fz, flag_zones=nz,
                                    false_ml=float(((um >= t) & ~em).sum() * np.prod(sp) / 1000)))
    print(f"{case} fatto", flush=True)
    return comp_rows, fa_rows


def curve(comp, fa, grid, rule, axis):
    """Per una misura e una fascia: frazione di zone trovate e falsi allarmi medi per paziente, per soglia."""
    col = "u_p90" if rule == "10pct" else "u_max"
    n_cases = fa.case.nunique()
    found = np.array([(comp[col] >= t).mean() if len(comp) else np.nan for t in grid])
    fcol = "false_zones" if axis == "zones" else "false_ml"
    fz = fa.groupby("t")[fcol].sum().reindex(grid).values / n_cases
    return found, fz


def at_fa(found, fz, target):
    """Zone trovate al livello di falsi allarmi `target`: massimo fra le soglie con falsi allarmi <= target."""
    ok = fz <= target
    return float(np.nanmax(found[ok])) if ok.any() else 0.0


def summarize(comp, fa, grid, out_dir, n_boot=2000, seed=0):
    """Curve e punti operativi. Bootstrap vettoriale: per ogni caso si precalcolano, per soglia, le zone
    trovate e i falsi allarmi; un campione bootstrap e' un vettore di molteplicita' dei casi."""
    rng = np.random.default_rng(seed)
    rows, curves = [], []
    for excl in [False, True]:
        tag = "noMass" if excl else "all"
        c0 = comp[~comp.case.isin(MASS)] if excl else comp
        f0 = fa[~fa.case.isin(MASS)] if excl else fa
        cases = np.array(sorted(f0.case.unique()))
        ci = {c: i for i, c in enumerate(cases)}
        W = rng.multinomial(len(cases), np.ones(len(cases)) / len(cases), size=n_boot)  # molteplicita'
        for (ut, w), f in f0.groupby(["uncertainty_type", "w"]):
            c = c0[(c0.uncertainty_type == ut) & (c0.w == w)]
            axes = [("10pct", "volume"), ("any", "volume")]
            if f.false_zones.notna().all():
                axes += [("10pct", "zones"), ("any", "zones")]
            ncomp = np.zeros(len(cases))
            for cs_, n in c.case.value_counts().items():
                ncomp[ci[cs_]] = n
            for rule, axis in axes:
                col = "u_p90" if rule == "10pct" else "u_max"
                fcol = "false_zones" if axis == "zones" else "false_ml"
                Fz = f.pivot(index="case", columns="t", values=fcol).reindex(index=cases, columns=grid).values
                Fc = np.zeros((len(cases), len(grid)))   # zone trovate per caso e soglia
                for cs_, d in c.groupby("case"):
                    Fc[ci[cs_]] = (d[col].values[:, None] >= grid[None, :]).sum(0)
                found = Fc.sum(0) / max(ncomp.sum(), 1)
                fz = Fz.sum(0) / len(cases)
                curves.append(pd.DataFrame(dict(subset=tag, uncertainty_type=ut, w=w, rule=rule, axis=axis, t=grid,
                                                found_frac=found, false_per_case=fz)))
                bf = (W @ Fc) / np.maximum(W @ ncomp, 1)[:, None]
                bz = (W @ Fz) / len(cases)
                for target in FA_POINTS[axis]:
                    val = at_fa(found, fz, target)
                    boots = [at_fa(bf[b], bz[b], target) for b in range(n_boot)]
                    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
                    rows.append(dict(subset=tag, uncertainty_type=ut, w=w, rule=rule, axis=axis, fa_per_case=target,
                                     found_frac=val, ci_lo=lo, ci_hi=hi, n_zones=len(c), n_cases=len(cases)))
    pd.concat(curves).to_csv(os.path.join(out_dir, "froc_curves.csv"), index=False)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, "froc_operating_points.csv"), index=False)
    lines = []
    for (tag, axis, rule, fa_), d in res.groupby(["subset", "axis", "rule", "fa_per_case"]):
        t = d.pivot(index="uncertainty_type", columns="w", values="found_frac")
        t = t.sort_values(t.columns[0], ascending=False)
        unit = "zone false" if axis == "zones" else "ml falsi"
        lines += [f"\n### {tag} | regola {rule} | zone trovate con <= {fa_:g} {unit} per paziente", t.round(3).to_string()]
    txt = "\n".join(lines)
    open(os.path.join(out_dir, "summary.txt"), "w").write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_dir", default="/mnt/sda2/grolo/kidney/predictions/deep")
    ap.add_argument("--gt_dir", default="/mnt/sda2/grolo/nnUNet/raw/Dataset904_KidneyTSMechanical/labelsTs")
    ap.add_argument("--unc_dir", default="/mnt/sda2/grolo/kidney/deep/kidney_uncertainty_maps")
    ap.add_argument("--unc_types", required=True)
    ap.add_argument("--border_mm", default="2,3,4.5")
    ap.add_argument("--min_vox", type=int, default=10)
    ap.add_argument("--cases", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--with_zones", action="store_true", help="calcola anche l'asse 'zone false' (lento)")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    a.widths = [float(w) for w in a.border_mm.split(",")]
    # griglia fitta in basso (le misure di disaccordo hanno tau* ~0.02-0.15), piu' rada in alto
    a.grid = np.unique(np.round(np.concatenate([np.arange(0.005, 0.05, 0.005), np.arange(0.05, 0.2, 0.025),
                                                np.arange(0.2, 1.0, 0.05)]), 4))
    os.makedirs(a.out_dir, exist_ok=True)
    cases = sorted(f[:-7] for f in os.listdir(a.gt_dir) if f.endswith(".nii.gz"))
    if a.cases:
        cases = [c for c in cases if c in a.cases.split(",")]
    measures = a.unc_types.split(",")
    with Pool(a.workers) as pool:
        res = pool.map(process_case, [(c, measures, a) for c in cases])
    comp = pd.DataFrame([r for x in res for r in x[0]])
    fa = pd.DataFrame([r for x in res for r in x[1]])
    comp.to_csv(os.path.join(a.out_dir, "zones_scores.csv"), index=False)
    fa.to_csv(os.path.join(a.out_dir, "false_zones_by_threshold.csv"), index=False)
    summarize(comp, fa, a.grid, a.out_dir)


if __name__ == "__main__":
    main()
