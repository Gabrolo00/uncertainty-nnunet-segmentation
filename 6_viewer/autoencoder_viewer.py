"""
Viewer dell'autoencoder: CT, errori, mappa di incertezza, ricostruzione, residui, combinazioni e misure di
riferimento, con il budget di allarmi dell'1%. Ogni caso e' mostrato con l'autoencoder che non l'ha visto.

Uso: streamlit run autoencoder_viewer.py --server.port 8502
"""
import importlib
import json
import os
import sys

sys.path[:0] = ['/home/grolo/thesis_pipeline/5_autoencoder']  # cartelle dei moduli del progetto importati qui

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from scipy.ndimage import distance_transform_edt  # noqa: E402
from skimage.segmentation import find_boundaries  # noqa: E402

import autoencoder_config  # noqa: E402

st.set_page_config(layout="wide", page_title="Autoencoder")
SOURCES = {"totalseg": "internal test set", "humanitas": "external test set"}
# Solo il modello della tesi (904), nomi come nel viewer principale
DATASETS = {"Kidney — deep ensemble": ("904", "totalseg"),
            "Kidney — deep ensemble, external test set": ("904", "humanitas")}
st.sidebar.title("Autoencoder Viewer")
ds_label = st.sidebar.selectbox("📁 Dataset / model", list(DATASETS))
DS, src = DATASETS[ds_label]
if os.environ.get("AE_DATASET") != DS or "autoencoder_eval" not in sys.modules:
    # la configurazione si legge all'import: al cambio di dataset si ricaricano i moduli
    os.environ["AE_DATASET"] = DS
    for mod in ["autoencoder_config", "autoencoder_train_infer", "autoencoder_eval", "autoencoder_stats", "autoencoder_combine"]:
        if mod in sys.modules:
            importlib.reload(sys.modules[mod])
import autoencoder_eval as E  # noqa: E402
from autoencoder_combine import combos  # noqa: E402

D, R = E.D, E.R
VARIANT_LABELS = {"a": "(a) entropy", "b": "(b) entropy + logit", "c": "(c) entropy + logit + CT"}
RUN_SUFFIX = {"All cases": "", "Well-segmented cases only": "_good", "Random subset (control)": "_rand"}
RUN_HELP = ("Cases used to train the autoencoder. All cases: about 165 cases. Well-segmented cases only: internal "
            "cases with less than 0.5 mL of errors more than 2 mm from the predicted boundary. Random subset: the "
            "same number of cases drawn at random, as a control. The last two exist only for variant (a).")
FAR_W = 2.0
# Colori del viewer classico: GT rosso, predizione viola, FP arancio, FN azzurro; seno renale grigio
GT_COLOR, PRED_COLOR, FP_COLOR, FN_COLOR, SINUS_COLOR = "#ff0000", "#ff00ff", "#f97316", "#38bdf8", "#9a9a9a"
CT_LEVEL, CT_WIDTH = 40, 370
ORDER_LABELS = {"AUPRCn obiettivo2 entropy": "Entropy: normalized AUPRC, target view (2 mm)",
                "AUPRCn obiettivo2 EBCU": "EBCU: normalized AUPRC, target view (2 mm)",
                "AUPRCn obiettivo2 a:excess": "Excess (a): normalized AUPRC, target view (2 mm)",
                "errori lontani mL": "Volume of errors away from the boundary (mL)"}


# ------------------------------------------------------------------ dati
@st.cache_data
def load_folds(ds, labels):   # labels nella chiave della cache: se cambiano i nomi, la cache si rinnova
    f = pd.read_csv(f"{D}/folds.csv").set_index("case")
    ev = f"{E.OUT}/eval_per_case.csv"
    if os.path.exists(ev):
        d = pd.read_csv(ev)
        d = d[(d.view == "farVSall2") & d["map"].isin(["entropy", "EBCU", "a:excess"])]
        t = d.pivot_table(index="case", columns="map", values="auprc_norm").round(3)
        n = d.drop_duplicates("case").set_index("case").n_pos * 1.5 ** 3 / 1000
        f = f.join(t.add_prefix("AUPRCn obiettivo2 ")).join(n.rename("errori lontani mL").round(2))
    return f.rename(columns=dict(labels))


@st.cache_data(max_entries=2)
def load_base(ds, c):
    pr = np.load(f"{D}/{c}/pred.npy").astype(bool)
    gt = np.load(f"{D}/{c}/gt.npy").astype(bool)
    ct = np.load(f"{D}/{c}/ct.npy").astype(np.float32)
    ad = np.abs(np.load(f"{D}/{c}/dist.npy").astype(np.float32))
    sinus = E.sinus_fn(c, pr, gt)
    roi = (distance_transform_edt(~(pr | gt), sampling=1.5) <= 15) & ~sinus
    lab = np.zeros(pr.shape, np.uint8)   # 0 niente, 1-4 errori, 5 seno
    far = ad > FAR_W
    lab[pr & ~gt & roi & ~far] = 1
    lab[pr & ~gt & roi & far] = 2
    lab[~pr & gt & roi & ~far] = 3
    lab[~pr & gt & roi & far] = 4
    lab[sinus] = 5
    return ct, pr, gt, lab, roi


@st.cache_data(max_entries=3)
def load_maps(ds, c, fold, variant, suffix):
    pr = np.load(f"{D}/{c}/pred.npy").astype(bool)
    refs = E.reference_maps(c, pr)
    u = refs["entropy"]
    maps = {"Entropy U": u, "Mutual information": refs["mi"], "DEU": refs["DEU"], "EBCU": refs["EBCU"]}
    md = f"{R}/{E.VARIANTS[variant]}{suffix}/fold{fold}/maps/{c}"
    if os.path.exists(f"{md}/recon.npy"):
        rec = np.load(f"{md}/recon.npy").astype(np.float32)
        maps.update({"Reconstruction Û": rec, "Excess": np.maximum(u - rec, 0),
                     "Lack": np.maximum(rec - u, 0), "Excess + lack": np.abs(u - rec)})
        rl = None
        for ch in ["logit", "ct"]:
            if os.path.exists(f"{md}/resid_{ch}.npy"):
                maps[{"logit": "Logit residual", "ct": "CT residual"}[ch]] = np.load(f"{md}/resid_{ch}.npy").astype(np.float32)
        if "Logit residual" in maps:
            rl = maps["Logit residual"]
        # parametri scelti per quella run: combine/ per le run complete, combine_good/ per casi buoni e controllo
        cp = f"{E.OUT}/combine{'_good' if suffix else ''}/chosen_params.json"
        key = {"": variant, "_good": "a_good", "_rand": "a_rand"}[suffix]
        if os.path.exists(cp):
            maps.update({f"Combination {k.split('(')[0]}": m for k, m in combos(u, rec, rl, json.load(open(cp))[key]).items()})
    return maps, md


def sl(vol, plane, i):
    # assiale ruotata di 180 gradi: anteriore in alto, destra del paziente a sinistra (convenzione radiologica)
    return {"Axial": vol[i][::-1, ::-1], "Coronal": vol[:, i, :], "Sagittal": vol[:, :, i]}[plane]


# ------------------------------------------------------------------ sidebar
folds = load_folds(DS, tuple(ORDER_LABELS.items()))
st.sidebar.caption(f"T* = {E.C['T_STAR']:.3f} · autoencoder with 4 stages and a 16-channel bottleneck")
cases = folds[folds.source == src]
order = st.sidebar.selectbox("Order the cases by", ["Case name"] + [c for c in ORDER_LABELS.values() if c in folds],
                             help="Puts on top the cases with the highest value, e.g. those with more errors away from the "
                                  "boundary or those in which a map works better. Normalized AUPRC in the target view: "
                                  "0 = random map, 1 = perfect.")
cases = cases.sort_index() if order == "Case name" else cases.sort_values(order, ascending=False)
case = st.sidebar.selectbox("Case", list(cases.index))
fold = int(folds.loc[case, "fold"])
variant = st.sidebar.radio("Input variant", list(VARIANT_LABELS), format_func=VARIANT_LABELS.get)
runs = [k for k, s in RUN_SUFFIX.items() if os.path.isdir(f"{R}/{E.VARIANTS[variant]}{s}/fold{fold}/maps/{case}")]
if not runs:
    st.error(f"No autoencoder maps for {case} (variant {variant}, fold {fold})")
    st.stop()
if len(runs) > 1:
    run = st.sidebar.radio("Training cases", runs, help=RUN_HELP)
else:
    run = runs[0]
    st.sidebar.caption(f"Training cases: {run.lower()}")

ct, pr, gt, lab, roi = load_base(DS, case)
maps, md = load_maps(DS, case, fold, variant, RUN_SUFFIX[run])
plane = st.sidebar.radio("View Plane", ["Axial", "Coronal", "Sagittal"], horizontal=True)
ax_n = {"Axial": 0, "Coronal": 1, "Sagittal": 2}[plane]
far_err = np.isin(lab, [2, 4])
counts = far_err.sum(axis=tuple(a for a in range(3) if a != ax_n))
default = int(counts.argmax()) if counts.max() > 0 else ct.shape[ax_n] // 2
if st.session_state.get("slice_ctx") != (case, plane):   # nuovo caso o piano: slice con piu' errori lontani
    st.session_state["slice_ctx"] = (case, plane)
    st.session_state["slice"] = default


def _go_default():
    st.session_state["slice"] = default


st.sidebar.button("Jump to the slice with most errors away from the boundary", on_click=_go_default)
idx = st.sidebar.slider("Slice Index", 0, ct.shape[ax_n] - 1, key="slice")

st.sidebar.markdown("---")
st.sidebar.subheader("Overlay Settings")
show_gt = st.sidebar.checkbox("Show Reference", True)
show_pr = st.sidebar.checkbox("Show Prediction", True)
show_fp = st.sidebar.checkbox("Show False Positives", True, help="Voxels predicted as kidney that are not in the reference.")
show_fn = st.sidebar.checkbox("Show False Negatives", True, help="Voxels of the reference that are not predicted.")
err_zone = st.sidebar.radio("Errors shown", ["Only away from the boundary (> 2 mm)", "All"],
                            help="Away from the boundary = more than 2 mm from the predicted boundary.")
show_sinus = st.sidebar.checkbox("Show Excluded Renal Sinus", False,
                                 help="False negatives inside the predicted kidney (sinus and hilum), excluded from "
                                      "the evaluation because of the different annotation convention.")
st.sidebar.markdown("---")
st.sidebar.subheader("Maps")
names = list(maps)
compare = st.sidebar.checkbox("Two maps side by side", True)
m1 = st.sidebar.selectbox("Map 1", names, index=names.index("Entropy U"))
m2 = st.sidebar.selectbox("Map 2", names, index=names.index("Excess") if "Excess" in names else 0) \
    if compare else None
alpha = st.sidebar.slider("Map opacity", 0.0, 1.0, 0.6, 0.05)
top_only = st.sidebar.checkbox("Show only the alarms (top 1% of the voxels)", False,
                               help="Shows only the voxels with the highest values, up to 1% of the region of interest "
                                    "(about 10 mL per case), the same budget for all the maps, to see where each map "
                                    "places its alarms: along the boundary or on the errors away from it.")
wl = (CT_LEVEL - CT_WIDTH / 2, CT_LEVEL + CT_WIDTH / 2)   # finestra fissa come nel viewer classico


# ------------------------------------------------------------------ disegno
def draw(name):
    mp = maps[name]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(sl(ct, plane, idx), cmap="gray", vmin=wl[0], vmax=wl[1], origin="lower" if ax_n else "upper")
    s = sl(mp, plane, idx).astype(np.float32)
    vals = np.nan_to_num(mp[roi])
    vmax = max(float(np.percentile(vals, 99.5)), 1e-6)
    if top_only:
        thr = np.quantile(vals, 0.99)
        s = np.where(s >= max(thr, 1e-12), s, np.nan)
    else:
        s = np.where(s > vmax * 0.02, s, np.nan)
    im = ax.imshow(s, cmap="viridis", vmin=0, vmax=vmax, alpha=alpha, origin="lower" if ax_n else "upper")
    for on, mask, col in [(show_gt, gt, GT_COLOR), (show_pr, pr, PRED_COLOR)]:
        if on:
            b = find_boundaries(sl(mask, plane, idx), mode="thick")
            ax.imshow(np.where(b, 1, np.nan), cmap=ListedColormap([col]), origin="lower" if ax_n else "upper")
    lsl = sl(lab, plane, idx)
    far_only = err_zone.startswith("Only")
    layers = [(show_fp, [2] if far_only else [1, 2], FP_COLOR), (show_fn, [4] if far_only else [3, 4], FN_COLOR),
              (show_sinus, [5], SINUS_COLOR)]
    for on, codes, col in layers:
        if on:
            ax.imshow(np.where(np.isin(lsl, codes), 1, np.nan), cmap=ListedColormap([col]), alpha=0.5,
                      origin="lower" if ax_n else "upper")
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    suf = " (> 2 mm)" if far_only else ""
    handles = [Line2D([], [], color=c_, lw=2, label=l_) for on, c_, l_ in
               [(show_gt, GT_COLOR, "Reference"), (show_pr, PRED_COLOR, "Prediction")] if on]
    handles += [mpatches.Patch(color=c_, alpha=0.5, label=l_) for on, c_, l_ in
                [(show_fp, FP_COLOR, "False positives" + suf), (show_fn, FN_COLOR, "False negatives" + suf),
                 (show_sinus, SINUS_COLOR, "Renal sinus (excluded)")] if on]
    if handles:
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=8, frameon=False)
    ax.set_title(f"{name}  (scale 0-{vmax:.3g})", fontsize=10)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.035)
    return fig


st.markdown(f"**{case}** ({SOURCES[src]}) — variant {VARIANT_LABELS[variant]} — training cases: {run.lower()}")
info = folds.loc[[case]].drop(columns=["source", "fold", "gt_source"], errors="ignore")
info = info.rename(columns={ORDER_LABELS["AUPRCn obiettivo2 entropy"]: "Entropy (AUPRC norm., target 2 mm)",
                            ORDER_LABELS["AUPRCn obiettivo2 EBCU"]: "EBCU (AUPRC norm., target 2 mm)",
                            ORDER_LABELS["AUPRCn obiettivo2 a:excess"]: "Excess (a) (AUPRC norm., target 2 mm)",
                            ORDER_LABELS["errori lontani mL"]: "Errors > 2 mm (mL)"})
st.dataframe(info, use_container_width=True)
cols = st.columns(2 if compare else 1)
for col, name in zip(cols, [m1, m2] if compare else [m1]):
    with col:
        st.pyplot(draw(name))
st.caption("Away from the boundary = more than 2 mm from the predicted boundary. Renal sinus = false negatives inside "
           "the predicted kidney, excluded from the evaluation. Each case is shown with the autoencoder that was not "
           "trained on it. Color scale of each map: from 0 to the 99.5th percentile of the map in the region of interest. "
           "CT with abdominal window (level 40 HU, width 370 HU), as in the uncertainty viewer.")
