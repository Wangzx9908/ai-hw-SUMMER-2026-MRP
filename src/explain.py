"""Stage 2: explain the IDS decisions with SHAP (global + local).

Everything here is computed on the official KDDTest+ split, i.e. on traffic the
model has never seen, which is the situation a network engineer actually faces.

Outputs
-------
figures/global_per_class.png   per-class global attribution (protocol-level)
figures/beeswarm_dos.png       direction + value of each field for the DoS head
figures/local_neptune.png      one blocked SYN-flood flow, waterfall
figures/local_guess_passwd.png one MISSED password-guessing flow (debug case)
results/shap_global.json       numeric tables reused by the slides
results/reason_strings.txt     attributions rendered as firewall-style reasons
"""

from __future__ import annotations

import json
import pathlib

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

import dataset as ds

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG, RESULTS = ROOT / "figures", ROOT / "results"
N_EXPLAIN = 3000          # test flows attributed with exact TreeSHAP
SEED = 0


def display_matrix(X: np.ndarray, b: ds.Bundle) -> np.ndarray:
    """(n, 122) one-hot -> (n, 41) values suitable for plot colouring.

    Numeric fields keep their value; a categorical field becomes the ordinal
    index of its active level (colour only, never used for maths).
    """
    out = np.zeros((X.shape[0], len(b.groups)), dtype=np.float32)
    for i, idx in enumerate(b.groups.values()):
        out[:, i] = X[:, idx[0]] if len(idx) == 1 else X[:, idx].argmax(axis=1)
    return out


def category_of(x_row: np.ndarray, field: str, b: ds.Bundle) -> str:
    idx = b.groups[field]
    return b.columns[idx[int(x_row[idx].argmax())]].split("=", 1)[1]


def pretty_value(x_row: np.ndarray, field: str, b: ds.Bundle) -> str:
    idx = b.groups[field]
    if len(idx) > 1:
        return category_of(x_row, field, b)
    v = float(x_row[idx[0]])
    return f"{v:.0f}" if abs(v - round(v)) < 1e-6 else f"{v:.2f}"


def reason_string(agg_row: np.ndarray, x_row: np.ndarray, b: ds.Bundle,
                  cls: str, prob: float, top: int = 4) -> str:
    """Render a local attribution the way a firewall log line should read."""
    order = np.argsort(-agg_row)[:top]
    parts = [f"{b.semantic_fields[j]}={pretty_value(x_row, b.semantic_fields[j], b)} "
             f"(+{agg_row[j]:.3f})" for j in order if agg_row[j] > 0]
    return f"verdict={cls} p={prob:.2f} because " + ", ".join(parts)


def display_row_strings(x_row: np.ndarray, b: ds.Bundle) -> np.ndarray:
    """Label values for a single flow: categoricals as their level name.

    Avoids waterfall labels like "flag = 1", which mean nothing to an engineer;
    they become "flag = S0" instead.
    """
    return np.array([pretty_value(x_row, f, b) for f in b.semantic_fields],
                    dtype=object)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    b = ds.load()
    rf = joblib.load(RESULTS / "random_forest.joblib")

    # ---- exact TreeSHAP on a stratified sample of the official test set ----
    sel = []
    for c in range(len(ds.CLASSES)):
        pool = np.flatnonzero(b.y_test == c)
        sel.append(rng.choice(pool, size=min(len(pool), N_EXPLAIN // 5),
                              replace=False))
    sel = np.sort(np.concatenate(sel))
    Xs, ys = b.X_test[sel], b.y_test[sel]

    explainer = shap.TreeExplainer(rf)
    sv = explainer.shap_values(Xs, check_additivity=False)   # (n, 122, 5)
    agg = ds.aggregate_to_fields(np.asarray(sv), b.groups, axis=1)  # (n, 41, 5)
    Xd = display_matrix(Xs, b)
    fields = b.semantic_fields

    # ---------------- global, per attack family ----------------
    tables = {}
    for k, cname in enumerate(ds.CLASSES):
        m = ys == k
        imp = np.abs(agg[m, :, k]).mean(axis=0)
        order = np.argsort(-imp)[:10]
        tables[cname] = [{"field": fields[j], "mean_abs_shap": round(float(imp[j]), 5)}
                         for j in order]

    panel = ["DoS", "Probe", "R2L"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, cname in zip(axes, panel):
        rows = tables[cname][::-1]
        ax.barh([r["field"] for r in rows], [r["mean_abs_shap"] for r in rows],
                color="#4a6fa5")
        ax.set_title(f"{cname} head", fontsize=11)
        ax.set_xlabel("mean |SHAP|", fontsize=9)
        ax.tick_params(labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Global attribution on unseen KDDTest+ traffic "
                 "(41 protocol fields, one-hot groups summed)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "global_per_class.png", dpi=170)
    plt.close(fig)

    # ---------------- beeswarm for the DoS head ----------------
    dos = ds.CLASSES.index("DoS")
    swarm_names = [f + (" *" if len(b.groups[f]) > 1 else "") for f in fields]
    expl_dos = shap.Explanation(
        values=agg[:, :, dos], base_values=np.full(agg.shape[0], 0.0),
        data=Xd, feature_names=swarm_names)
    plt.figure()
    shap.plots.beeswarm(expl_dos, max_display=12, show=False)
    plt.title("Direction of each field for the DoS output\n"
              "(* categorical, coloured by level index)", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG / "beeswarm_dos.png", dpi=170)
    plt.close("all")

    # ---------------- local: a correctly blocked SYN flood ----------------
    proba = rf.predict_proba(Xs)
    names = b.attack_test[sel]
    reasons: list[str] = []

    def local_plot(mask: np.ndarray, target: int, fname: str, title: str) -> int | None:
        cand = np.flatnonzero(mask)
        if cand.size == 0:
            return None
        i = cand[np.argmax(proba[cand, target])]
        e = shap.Explanation(values=agg[i, :, target],
                             base_values=float(explainer.expected_value[target]),
                             data=display_row_strings(Xs[i], b),
                             feature_names=fields)
        plt.figure()
        shap.plots.waterfall(e, max_display=9, show=False)
        plt.title(title, fontsize=10)
        plt.tight_layout()
        plt.savefig(FIG / fname, dpi=170)
        plt.close("all")
        return int(i)

    i_dos = local_plot(
        (names == "neptune") & (rf.predict(Xs) == dos), dos, "local_neptune.png",
        "Local explanation: neptune (SYN flood) correctly blocked as DoS")
    if i_dos is not None:
        reasons.append("[TP / DoS] " + reason_string(
            agg[i_dos, :, dos], Xs[i_dos], b, "DoS", float(proba[i_dos, dos])))

    # ---------------- local: a MISSED R2L attack (debug case) ----------------
    r2l, normal = ds.CLASSES.index("R2L"), ds.CLASSES.index("normal")
    miss = (np.isin(names, ["guess_passwd", "warezmaster", "snmpguess"])
            & (rf.predict(Xs) == normal))
    i_fn = local_plot(miss, normal, "local_guess_passwd.png",
                      "Debug case: R2L attack waved through as normal "
                      "- which fields caused it?")
    if i_fn is not None:
        reasons.append(f"[FN / {names[i_fn]}] " + reason_string(
            agg[i_fn, :, normal], Xs[i_fn], b, "normal",
            float(proba[i_fn, normal])))

    # a couple more reason strings for the appendix slide
    probe = ds.CLASSES.index("Probe")
    for atk, tgt, tag in [("portsweep", probe, "TP / Probe"),
                          ("satan", probe, "TP / Probe")]:
        cand = np.flatnonzero((names == atk) & (rf.predict(Xs) == tgt))
        if cand.size:
            i = cand[np.argmax(proba[cand, tgt])]
            reasons.append(f"[{tag}] " + reason_string(
                agg[i, :, tgt], Xs[i], b, ds.CLASSES[tgt], float(proba[i, tgt])))

    (RESULTS / "reason_strings.txt").write_text("\n".join(reasons) + "\n")
    (RESULTS / "shap_global.json").write_text(json.dumps(
        {"n_explained": int(Xs.shape[0]), "per_class_top10": tables}, indent=2))

    np.savez_compressed(RESULTS / "shap_cache.npz", sel=sel, agg=agg,
                        Xd=Xd, ys=ys)
    print(f"explained {Xs.shape[0]} unseen flows; figures in {FIG}")
    for r in reasons:
        print("  " + r)


if __name__ == "__main__":
    main()
