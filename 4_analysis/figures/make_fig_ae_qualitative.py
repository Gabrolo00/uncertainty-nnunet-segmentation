"""
Figure qualitative dell'autoencoder: mappe residue su una fetta (residual) e allarmi delle mappe con lo stesso
budget dell'1% (alarms).

Uso: AE_DATASET=904 python make_fig_ae_qualitative.py residual|alarms <caso> [fetta|-] [out.png]
"""
import json
import os
import sys

sys.path[:0] = ['/home/grolo/thesis_pipeline/5_autoencoder']  # cartelle dei moduli del progetto importati qui
os.environ.setdefault("AE_DATASET", "904")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

import autoencoder_eval as E  # noqa: E402
from autoencoder_combine import combos  # noqa: E402

kind, c = sys.argv[1], sys.argv[2]
z_arg = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != "-" else None
out = sys.argv[4] if len(sys.argv) > 4 else f"draft_ae_{kind}_{c}.png"
VAR, FAR, TOP = "c", 2.0, 0.01
FP_C, FN_C, REF_C, PRED_C = (1, 0.55, 0, 1), (0.1, 0.6, 1, 1), "#00c000", "#ff3030"   # come fig. del cap. 5

D, R = E.D, E.R
fold = int(pd.read_csv(f"{D}/folds.csv").set_index("case").loc[c, "fold"])
pr = np.load(f"{D}/{c}/pred.npy").astype(bool)
gt = np.load(f"{D}/{c}/gt.npy").astype(bool)
ct = np.load(f"{D}/{c}/ct.npy").astype(np.float32)
ad = np.abs(np.load(f"{D}/{c}/dist.npy").astype(np.float32))
sinus = E.sinus_fn(c, pr, gt)
roi = (ndi.distance_transform_edt(~(pr | gt), sampling=1.5) <= 15) & ~sinus
far_err = (pr ^ gt) & roi & (ad > FAR)
refs = E.reference_maps(c, pr)
u = refs["entropy"]
md = f"{R}/{E.VARIANTS[VAR]}/fold{fold}/maps/{c}"
rec = np.load(f"{md}/recon.npy").astype(np.float32)
rl = np.load(f"{md}/resid_logit.npy").astype(np.float32)
excess = np.maximum(u - rec, 0)
cmb = combos(u, rec, rl, json.load(open(f"{E.OUT}/combine/chosen_params.json"))[VAR])
c4 = [v for k, v in cmb.items() if k.startswith("C4")][0]

z = z_arg if z_arg is not None else int(far_err.sum(axis=(1, 2)).argmax())
print(c, "fold", fold, "slice", z, "far error voxels on slice", int(far_err[z].sum()),
      "far error mL", round(far_err.sum() * 1.5 ** 3 / 1000, 2))

# ritaglio quadrato attorno alla componente della ROI con gli errori lontani della fetta
roiz = ndi.binary_dilation(pr[z] | gt[z], iterations=6)
lab, _ = ndi.label(roiz)
k_id = np.bincount(lab[far_err[z]].ravel()).argmax() if far_err[z].any() else 1
ys, xs = np.where(lab == k_id)
cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
half = max(ys.max() - ys.min(), xs.max() - xs.min()) // 2 + 2
y0, x0 = max(cy - half, 0), max(cx - half, 0)
cr = lambda a: a[z][y0:cy + half, x0:cx + half]
ctw = np.clip((cr(ct) + 145) / 370, 0, 1)   # finestra L40 / W370, come i viewer
g, p, fe = cr(gt), cr(pr), cr(far_err)


def base(a, errors=True):
    a.imshow(ctw, cmap="gray", vmin=0, vmax=1)
    if errors:
        e = np.zeros(g.shape + (4,))
        e[fe & p & ~g] = FP_C
        e[fe & g & ~p] = FN_C
        a.imshow(e, alpha=0.85)
    a.contour(g, [0.5], colors=[REF_C], linewidths=0.9)
    a.contour(p, [0.5], colors=[PRED_C], linewidths=0.8)
    a.set_xticks([]); a.set_yticks([])


def legend(fig, extra=()):
    h = [Line2D([], [], color=REF_C, label="Reference"), Line2D([], [], color=PRED_C, label="Prediction"),
         Patch(color=FP_C, label="False positive ($>2$ mm)"), Patch(color=FN_C, label="False negative ($>2$ mm)")]
    fig.legend(handles=h + list(extra), loc="lower left", ncol=len(h) + len(extra), fontsize=8, frameon=False,
               columnspacing=1.0, handlelength=1.4, bbox_to_anchor=(0.0, 0.0))


if kind == "residual":
    vmax = float(np.percentile(u[roi], 99.5))
    panels = [("$U$ (entropy)", u, vmax), (r"$\hat{U}$ (reconstruction)", rec, vmax), ("Excess", excess, vmax),
              ("EBCU", refs["EBCU"], 1.0)]
    fig = plt.figure(figsize=(10, 2.9))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 0.05], hspace=0.08, wspace=0.04, left=0.01, right=0.99, top=0.9,
                          bottom=0.17)
    a = fig.add_subplot(gs[0, 0]); base(a); a.set_title("Errors", fontsize=10)
    for j, (t, m, vm) in enumerate(panels):
        a = fig.add_subplot(gs[0, j + 1])
        im = a.imshow(np.nan_to_num(cr(m)), cmap="magma", vmin=0, vmax=vm)
        a.contour(g, [0.5], colors=[REF_C], linewidths=0.6)
        a.contour(fe, [0.5], colors=["white"], linewidths=0.6, linestyles="--")
        a.set_title(t, fontsize=10); a.set_xticks([]); a.set_yticks([])
        if j in (0, 3):
            cb = fig.colorbar(im, cax=fig.add_subplot(gs[1, j + 1]), orientation="horizontal")
            cb.ax.tick_params(labelsize=7)
    legend(fig, [Line2D([], [], color="0.4", ls="--", label="Errors $>2$ mm (outline)")])
    for t, m in [("U", u), ("rec", rec), ("excess", excess), ("EBCU", refs["EBCU"])]:
        print(f"  {t}: mean on far errors {np.nanmean(m[far_err]):.3f}, 99.5 pct ROI {np.percentile(np.nan_to_num(m[roi]), 99.5):.3f}")
else:
    rng = np.random.default_rng(0)
    panels = [("Entropy", u), ("Excess", excess), ("$C_4$", c4), ("EBCU", refs["EBCU"])]
    band = ad <= FAR
    fig = plt.figure(figsize=(9, 2.9))
    gs = fig.add_gridspec(1, 4, wspace=0.04, left=0.01, right=0.99, top=0.9, bottom=0.12)
    for j, (t, m) in enumerate(panels):
        s = np.nan_to_num(m[roi])
        K = int(round(TOP * s.size))
        order = np.lexsort((rng.random(s.size), -s))   # budget esatto: pareggi spezzati a caso
        top = np.zeros(s.size, bool); top[order[:K]] = True
        alarm = np.zeros(pr.shape, bool); alarm[roi] = top
        print(f"  {t}: alarms in 2 mm band {100 * (alarm & band).sum() / K:.0f}%, "
              f"far errors covered {100 * (alarm & far_err).sum() / max(far_err.sum(), 1):.1f}%")
        a = fig.add_subplot(gs[0, j]); base(a)
        ov = np.zeros(g.shape + (4,)); ov[cr(alarm)] = (1, 1, 0, 1)
        a.imshow(ov, alpha=0.75)
        a.contour(cr(band) & cr(roi), [0.5], colors=["white"], linewidths=0.5, linestyles=":")
        a.set_title(t, fontsize=10)
    legend(fig, [Patch(color=(1, 1, 0), label="Alarms (1% budget)"),
                 Line2D([], [], color="0.4", ls=":", label="2 mm band")])
fig.savefig(out, dpi=200, facecolor="white")
print("saved", out)
