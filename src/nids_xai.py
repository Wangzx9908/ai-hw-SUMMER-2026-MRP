"""XAI for Network Intrusion Detection - NSL-KDD + Random Forest + SHAP.

Run:  python3 nids_xai.py

Steps:
  1. load the NSL-KDD official train/test splits
  2. train a Random Forest to classify each flow as normal / DoS / Probe / R2L / U2R
  3. explain it with SHAP: one global importance plot + two local plots
"""

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA, FIG, RESULTS = ROOT / "data", ROOT / "figures", ROOT / "results"

# The 41 NSL-KDD flow features, in file order.
FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
CATEGORICAL = ["protocol_type", "service", "flag"]
CLASSES = ["normal", "DoS", "Probe", "R2L", "U2R"]

# Group the individual attack names into the 4 standard families.
FAMILY = {
    "neptune": "DoS", "back": "DoS", "land": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "mailbomb": "DoS", "apache2": "DoS",
    "processtable": "DoS", "udpstorm": "DoS", "worm": "DoS",
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L", "warezclient": "R2L",
    "warezmaster": "R2L", "sendmail": "R2L", "named": "R2L",
    "snmpgetattack": "R2L", "snmpguess": "R2L", "xlock": "R2L",
    "xsnoop": "R2L", "httptunnel": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
}


def load():
    """Read both splits and encode the 3 text columns as integers."""
    cols = FEATURES + ["label", "difficulty"]
    tr = pd.read_csv(DATA / "KDDTrain+.txt", names=cols)
    te = pd.read_csv(DATA / "KDDTest+.txt", names=cols)

    for df in (tr, te):
        df["family"] = df["label"].apply(
            lambda s: "normal" if s == "normal" else FAMILY.get(s, "unknown"))
    tr = tr[tr["family"] != "unknown"]
    te = te[te["family"] != "unknown"]

    # Integer codes for protocol_type / service / flag, shared across splits.
    # Tree models handle integer-coded categories fine, and it keeps the SHAP
    # plots readable: 41 features in, 41 feature names out.
    for c in CATEGORICAL:
        levels = sorted(set(tr[c]) | set(te[c]))
        mapping = {v: i for i, v in enumerate(levels)}
        tr[c] = tr[c].map(mapping)
        te[c] = te[c].map(mapping)

    X_train = tr[FEATURES].astype(float).to_numpy()
    X_test = te[FEATURES].astype(float).to_numpy()
    y_train = tr["family"].map(CLASSES.index).to_numpy()
    y_test = te["family"].map(CLASSES.index).to_numpy()
    return X_train, y_train, X_test, y_test, te["label"].to_numpy()


def main():
    FIG.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    X_train, y_train, X_test, y_test, attack_names = load()
    print(f"train {X_train.shape}, test {X_test.shape}")

    # ---------------- 1. train the detector ----------------
    rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=2,
                                class_weight="balanced_subsample",
                                n_jobs=-1, random_state=0)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"\naccuracy on the official test set: {acc:.3f}\n")
    print(classification_report(y_test, pred, labels=range(5),
                                target_names=CLASSES, zero_division=0))

    # ---------------- 2. SHAP values ----------------
    # A stratified sample keeps the run to a few seconds.
    rng = np.random.default_rng(0)
    idx = np.concatenate([
        rng.choice(np.flatnonzero(y_test == c),
                   size=min(400, int((y_test == c).sum())), replace=False)
        for c in range(5)])
    Xs = X_test[idx]

    explainer = shap.TreeExplainer(rf)
    sv = np.asarray(explainer.shap_values(Xs, check_additivity=False))
    print(f"SHAP values: {sv.shape}  (flows, features, classes)")

    # ---------------- 3. global explanation ----------------
    # Which features matter most for DoS and for Probe?
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, fam in zip(axes, ["DoS", "Probe"]):
        k = CLASSES.index(fam)
        m = y_test[idx] == k
        imp = np.abs(sv[m, :, k]).mean(axis=0)
        top = np.argsort(-imp)[:10][::-1]
        ax.barh([FEATURES[i] for i in top], imp[top], color="#4a6fa5")
        ax.set_title(f"{fam}: most important features", fontsize=11)
        ax.set_xlabel("mean |SHAP value|", fontsize=9)
        ax.tick_params(labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "shap_global.png", dpi=170)
    plt.close(fig)

    # ---------------- 4. local explanations ----------------
    proba = rf.predict_proba(Xs)
    names = attack_names[idx]
    summary = {}
    for attack, fam, fname in [("neptune", "DoS", "shap_local_dos.png"),
                               ("portsweep", "Probe", "shap_local_probe.png")]:
        k = CLASSES.index(fam)
        cand = np.flatnonzero((names == attack) & (pred[idx] == k))
        if cand.size == 0:
            continue
        i = cand[np.argmax(proba[cand, k])]
        e = shap.Explanation(values=sv[i, :, k],
                             base_values=float(explainer.expected_value[k]),
                             data=Xs[i], feature_names=FEATURES)
        plt.figure()
        shap.plots.waterfall(e, max_display=9, show=False)
        plt.title(f"Why this flow was flagged as {fam} ({attack})", fontsize=10)
        plt.tight_layout()
        plt.savefig(FIG / fname, dpi=170)
        plt.close("all")

        order = np.argsort(-sv[i, :, k])[:4]
        summary[attack] = {
            "predicted": fam,
            "probability": round(float(proba[i, k]), 3),
            "top_features": [
                {"feature": FEATURES[j], "value": round(float(Xs[i, j]), 2),
                 "shap": round(float(sv[i, j, k]), 3)} for j in order],
        }
        print(f"\n{attack} -> {fam} (p={proba[i, k]:.2f}) because "
              + ", ".join(f"{FEATURES[j]}={Xs[i, j]:g}" for j in order))

    # ---------------- 5. save the numbers ----------------
    (RESULTS / "results.json").write_text(json.dumps({
        "test_accuracy": round(float(acc), 4),
        "recall_per_class": {
            c: round(float(((pred == k) & (y_test == k)).sum()
                           / max((y_test == k).sum(), 1)), 3)
            for k, c in enumerate(CLASSES)},
        "top10_features": {
            fam: [FEATURES[i] for i in np.argsort(
                -np.abs(sv[y_test[idx] == CLASSES.index(fam), :,
                           CLASSES.index(fam)]).mean(axis=0))[:10]]
            for fam in ["DoS", "Probe", "R2L"]},
        "local_examples": summary,
    }, indent=2))
    print(f"\nfigures -> {FIG}\nnumbers -> {RESULTS / 'results.json'}")


if __name__ == "__main__":
    main()
