"""Stage 3: quantitative evaluation of the explanations themselves.

A pretty SHAP plot is not a result. Four questions are answered with numbers:

E1 protocol consistency  Does the attributed mass land on the fields that
                         network security textbooks associate with each attack
                         family, or somewhere arbitrary?
E2 faithfulness          If we neutralise the top-k attributed fields towards a
                         benign reference, does the attack score actually
                         collapse? SHAP vs LIME vs random ranking.
E3 stability             Same flow, same model, repeated explanations. LIME
                         samples randomly, TreeSHAP is exact - how large is the
                         gap in practice?
E4 cost                  ms per explained flow, i.e. can this run inline on a
                         firewall or only offline in a SOC console?
"""

from __future__ import annotations

import json
import pathlib
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr

import dataset as ds

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG, RESULTS = ROOT / "figures", ROOT / "results"
SEED = 0

# Standard NSL-KDD feature taxonomy (Tavallaee et al., 2009 / KDD'99 docs).
BASIC = ds.NUMERIC_AND_CAT_COLUMNS[0:9]
CONTENT = ds.NUMERIC_AND_CAT_COLUMNS[9:22]
TIME_TRAFFIC = ds.NUMERIC_AND_CAT_COLUMNS[22:31]
HOST_TRAFFIC = ds.NUMERIC_AND_CAT_COLUMNS[31:41]
TAXONOMY = {"basic": BASIC, "content": CONTENT,
            "time_traffic": TIME_TRAFFIC, "host_traffic": HOST_TRAFFIC}

# What a security engineer expects to matter, per family, before seeing any SHAP
# output. Fixed in advance so E1 is a test, not a post-hoc story.
EXPECTED = {
    "DoS":   ["time_traffic", "basic"],      # flood volume + half-open flags
    "Probe": ["time_traffic", "host_traffic"],  # fan-out over hosts/ports
    "R2L":   ["content", "basic"],           # payload / auth semantics
    "U2R":   ["content"],                    # shells, root, file creation
}


# ----------------------------------------------------------------- helpers
def field_of_column(b: ds.Bundle) -> np.ndarray:
    owner = np.empty(len(b.columns), dtype=np.int32)
    for i, idx in enumerate(b.groups.values()):
        owner[idx] = i
    return owner


def mask_fields(x: np.ndarray, field_ids, b: ds.Bundle) -> np.ndarray:
    """Move the given semantic fields to the benign reference profile."""
    out = x.copy()
    cols = [c for fid in field_ids for c in b.groups[b.semantic_fields[fid]]]
    out[cols] = b.background_row[cols]
    return out


def topk_jaccard(a: np.ndarray, c: np.ndarray, k: int = 5) -> float:
    sa, sc = set(np.argsort(-np.abs(a))[:k]), set(np.argsort(-np.abs(c))[:k])
    return len(sa & sc) / len(sa | sc)


def lime_field_weights(expl: LimeTabularExplainer, x: np.ndarray, predict,
                       label: int, b: ds.Bundle, owner: np.ndarray,
                       n_samples: int = 1000) -> np.ndarray:
    e = expl.explain_instance(x, predict, labels=(label,),
                              num_features=len(b.columns),
                              num_samples=n_samples)
    w = np.zeros(len(b.columns))
    for col, weight in e.as_map()[label]:
        w[col] = weight
    return np.bincount(owner, weights=w, minlength=len(b.groups))


# ----------------------------------------------------------------- experiments
def e1_protocol_consistency(agg: np.ndarray, ys: np.ndarray,
                            b: ds.Bundle) -> dict:
    idx_of = {f: i for i, f in enumerate(b.semantic_fields)}
    rows = {}
    for fam, expected in EXPECTED.items():
        k = ds.CLASSES.index(fam)
        m = ys == k
        if m.sum() == 0:
            continue
        imp = np.abs(agg[m, :, k]).mean(axis=0)
        total = imp.sum()
        share = {g: float(imp[[idx_of[f] for f in fs]].sum() / total)
                 for g, fs in TAXONOMY.items()}
        exp_share = sum(share[g] for g in expected)
        # uninformative baseline: mass proportional to the number of fields
        prior = sum(len(TAXONOMY[g]) for g in expected) / len(b.semantic_fields)
        rows[fam] = {
            "expected_groups": expected,
            "attributed_share_in_expected": round(exp_share, 3),
            "random_baseline_share": round(prior, 3),
            "lift": round(exp_share / prior, 2),
            "share_per_group": {g: round(v, 3) for g, v in share.items()},
        }
    return rows


def e2_faithfulness(rf, b: ds.Bundle, agg: np.ndarray, sel: np.ndarray,
                     owner: np.ndarray, n_flows: int = 120,
                     kmax: int = 10) -> dict:
    """Deletion test: neutralise top-k fields, watch the attack score fall."""
    rng = np.random.default_rng(SEED)
    proba = rf.predict_proba(b.X_test[sel])
    pred = proba.argmax(axis=1)
    cand = np.flatnonzero((pred == b.y_test[sel]) & (pred != 0))
    pick = rng.choice(cand, size=min(n_flows, cand.size), replace=False)

    lime_expl = LimeTabularExplainer(
        b.X_train, feature_names=b.columns, discretize_continuous=False,
        mode="classification", random_state=SEED)

    curves = {m: np.zeros(kmax + 1) for m in ("shap", "lime", "random")}
    for i in pick:
        x, lab = b.X_test[sel[i]], int(pred[i])
        p0 = float(proba[i, lab])
        rank = {
            "shap": np.argsort(-agg[i, :, lab]),
            "lime": np.argsort(-lime_field_weights(
                lime_expl, x, rf.predict_proba, lab, b, owner)),
            "random": rng.permutation(len(b.groups)),
        }
        for m, order in rank.items():
            batch = np.stack([mask_fields(x, order[:k], b)
                              for k in range(kmax + 1)])
            curves[m] += p0 - rf.predict_proba(batch)[:, lab]
    for m in curves:
        curves[m] /= len(pick)

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    style = {"shap": ("o-", "#1b4f72"), "lime": ("s--", "#c0392b"),
             "random": ("^:", "#7f8c8d")}
    for m, (mk, col) in style.items():
        ax.plot(range(kmax + 1), curves[m], mk, color=col, ms=4,
                label={"shap": "SHAP ranking", "lime": "LIME ranking",
                       "random": "random ranking"}[m])
    ax.set_xlabel("number of top-ranked fields neutralised (k)")
    ax.set_ylabel("drop in attack probability")
    ax.legend(frameon=False, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "faithfulness.png", dpi=170)
    plt.close(fig)

    return {"n_flows": int(len(pick)), "kmax": kmax,
            "curves": {m: [round(float(v), 4) for v in c]
                       for m, c in curves.items()},
            "aopc": {m: round(float(c[1:].mean()), 4) for m, c in curves.items()}}


def e3_e4_stability_and_cost(rf, mlp, b: ds.Bundle, agg: np.ndarray,
                             sel: np.ndarray, owner: np.ndarray,
                             n_flows: int = 25, repeats: int = 5) -> dict:
    rng = np.random.default_rng(SEED)
    pred = rf.predict(b.X_test[sel])
    cand = np.flatnonzero((pred == b.y_test[sel]) & (pred != 0))
    pick = rng.choice(cand, size=min(n_flows, cand.size), replace=False)

    # --- LIME: re-explain the same flow several times, different seeds
    lime_j, lime_ms = [], []
    for i in pick:
        runs = []
        for r in range(repeats):
            ex = LimeTabularExplainer(
                b.X_train, feature_names=b.columns, discretize_continuous=False,
                mode="classification", random_state=SEED + 100 * r)
            t0 = time.perf_counter()
            runs.append(lime_field_weights(ex, b.X_test[sel[i]],
                                           rf.predict_proba, int(pred[i]), b,
                                           owner))
            lime_ms.append((time.perf_counter() - t0) * 1e3)
        lime_j += [topk_jaccard(runs[a], runs[c])
                   for a in range(repeats) for c in range(a + 1, repeats)]

    # --- TreeSHAP: exact, so re-running must give bit-identical output
    tex = shap.TreeExplainer(rf)
    Xp = b.X_test[sel[pick]]
    t0 = time.perf_counter()
    v1 = np.asarray(tex.shap_values(Xp, check_additivity=False))
    tree_ms = (time.perf_counter() - t0) * 1e3 / len(pick)
    v2 = np.asarray(tex.shap_values(Xp, check_additivity=False))
    tree_j = float(np.mean([
        topk_jaccard(ds.aggregate_to_fields(v1[a], b.groups, axis=0)[:, int(pred[i])],
                     ds.aggregate_to_fields(v2[a], b.groups, axis=0)[:, int(pred[i])])
        for a, i in enumerate(pick)]))

    # --- KernelSHAP on the MLP: model-agnostic, the realistic DL case
    bg = shap.kmeans(b.X_train, 20)
    ksh = shap.KernelExplainer(mlp.predict_proba, bg)
    sub = Xp[:6]
    t0 = time.perf_counter()
    kv = ksh.shap_values(sub, nsamples=256, silent=True)
    kernel_ms = (time.perf_counter() - t0) * 1e3 / len(sub)

    # --- agreement between the two families on the same RF decisions
    agree = []
    lex = LimeTabularExplainer(b.X_train, feature_names=b.columns,
                               discretize_continuous=False,
                               mode="classification", random_state=SEED)
    rho = []
    for a, i in enumerate(pick):
        lab = int(pred[i])
        lw = lime_field_weights(lex, b.X_test[sel[i]], rf.predict_proba, lab, b,
                                owner)
        sw = agg[i, :, lab]
        agree.append(topk_jaccard(sw, lw))
        rho.append(spearmanr(sw, lw).statistic)

    return {
        "n_flows": int(len(pick)), "repeats": repeats,
        "stability_top5_jaccard": {"lime": round(float(np.mean(lime_j)), 3),
                                    "treeshap": round(tree_j, 3)},
        "cost_ms_per_flow": {
            "treeshap_rf": round(float(tree_ms), 2),
            "lime_rf_1000_samples": round(float(np.mean(lime_ms)), 1),
            "kernelshap_mlp_256_samples": round(float(kernel_ms), 1),
        },
        "shap_vs_lime": {"top5_jaccard": round(float(np.mean(agree)), 3),
                          "spearman_rho": round(float(np.nanmean(rho)), 3)},
        "kernelshap_output_shape": list(np.asarray(kv).shape),
    }


def main() -> None:
    b = ds.load()
    rf = joblib.load(RESULTS / "random_forest.joblib")
    mlp = joblib.load(RESULTS / "mlp.joblib")
    cache = np.load(RESULTS / "shap_cache.npz")
    sel, agg, ys = cache["sel"], cache["agg"], cache["ys"]
    owner = field_of_column(b)

    out = {"E1_protocol_consistency": e1_protocol_consistency(agg, ys, b)}
    print("E1 done")
    out["E2_faithfulness"] = e2_faithfulness(rf, b, agg, sel, owner)
    print("E2 done")
    out["E3_E4_stability_cost"] = e3_e4_stability_and_cost(
        rf, mlp, b, agg, sel, owner)
    print("E3/E4 done")

    (RESULTS / "xai_eval.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
