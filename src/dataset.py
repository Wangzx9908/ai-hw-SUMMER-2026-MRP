"""NSL-KDD loading / preprocessing for the XAI-NIDS mini research problem.

Design choice that matters for explainability
---------------------------------------------
Categorical fields (protocol_type, service, flag) must be one-hot encoded for
the models, which explodes 41 semantic fields into ~122 columns. Raw SHAP
attributions on those columns are unreadable for a network engineer
("service_ecr_i = 0.03"). We therefore keep a group map
    semantic field -> list of encoded column indices
so that SHAP values can be summed back to the 41 protocol-level fields
(Shapley values are additive, so summing over a disjoint group of features is
the exact attribution of that group).
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data"

NUMERIC_AND_CAT_COLUMNS = [
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
ALL_COLUMNS = NUMERIC_AND_CAT_COLUMNS + ["label", "difficulty"]
CATEGORICAL = ["protocol_type", "service", "flag"]

# Standard 4-family grouping of NSL-KDD attack names (train+ and test+ names).
ATTACK_FAMILY = {
    # Denial of Service
    "neptune": "DoS", "back": "DoS", "land": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "mailbomb": "DoS", "apache2": "DoS",
    "processtable": "DoS", "udpstorm": "DoS", "worm": "DoS",
    # Probe / reconnaissance
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    # Remote to Local
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L", "warezclient": "R2L",
    "warezmaster": "R2L", "sendmail": "R2L", "named": "R2L",
    "snmpgetattack": "R2L", "snmpguess": "R2L", "xlock": "R2L",
    "xsnoop": "R2L", "httptunnel": "R2L",
    # User to Root
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
}
CLASSES = ["normal", "DoS", "Probe", "R2L", "U2R"]


@dataclass
class Bundle:
    """Everything downstream scripts need, in one object."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    columns: list[str]              # ~122 encoded column names
    groups: dict[str, list[int]]    # 41 semantic fields -> encoded col indices
    semantic_fields: list[str]
    attack_train: np.ndarray        # raw attack name, for per-attack analysis
    attack_test: np.ndarray
    background_row: np.ndarray      # "benign reference" used for masking tests


def _read_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / name, names=ALL_COLUMNS, header=None)
    return df.drop(columns=["difficulty"])


def _family(label: str) -> str:
    return "normal" if label == "normal" else ATTACK_FAMILY.get(label, "unknown")


def load(drop_unknown: bool = True) -> Bundle:
    tr, te = _read_split("KDDTrain+.txt"), _read_split("KDDTest+.txt")

    for df in (tr, te):
        df["family"] = df["label"].map(_family)
    if drop_unknown:
        tr = tr[tr["family"] != "unknown"].reset_index(drop=True)
        te = te[te["family"] != "unknown"].reset_index(drop=True)

    attack_train = tr["label"].to_numpy()
    attack_test = te["label"].to_numpy()
    y_train = tr["family"].map(CLASSES.index).to_numpy()
    y_test = te["family"].map(CLASSES.index).to_numpy()

    tr = tr.drop(columns=["label", "family"])
    te = te.drop(columns=["label", "family"])

    # One-hot with a shared vocabulary taken from the training split only
    # (a test-only service must not create a brand new column).
    frames_tr, frames_te, groups, cols = [], [], {}, []
    for field in NUMERIC_AND_CAT_COLUMNS:
        if field in CATEGORICAL:
            vocab = sorted(tr[field].unique())
            enc_tr = pd.get_dummies(tr[field]).reindex(columns=vocab, fill_value=0)
            enc_te = pd.get_dummies(te[field]).reindex(columns=vocab, fill_value=0)
            names = [f"{field}={v}" for v in vocab]
        else:
            enc_tr, enc_te = tr[[field]], te[[field]]
            names = [field]
        groups[field] = list(range(len(cols), len(cols) + len(names)))
        cols.extend(names)
        frames_tr.append(np.asarray(enc_tr, dtype=np.float32))
        frames_te.append(np.asarray(enc_te, dtype=np.float32))

    X_train = np.hstack(frames_tr)
    X_test = np.hstack(frames_te)

    # Benign reference profile: median of the normal training traffic. Used as
    # the "what if this feature carried nothing suspicious" value in the
    # faithfulness (feature-masking) experiment.
    background_row = np.median(X_train[y_train == 0], axis=0).astype(np.float32)

    return Bundle(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        columns=cols, groups=groups,
        semantic_fields=list(NUMERIC_AND_CAT_COLUMNS),
        attack_train=attack_train, attack_test=attack_test,
        background_row=background_row,
    )


def aggregate_to_fields(values: np.ndarray, groups: dict[str, list[int]],
                        axis: int = -1) -> np.ndarray:
    """Sum attributions over one-hot groups along `axis`: n_cols -> n_fields.

    SHAP returns (n_samples, n_cols, n_classes) for multiclass models, so the
    feature axis is not always the last one.
    """
    parts = [values.take(idx, axis=axis).sum(axis=axis, keepdims=True)
             for idx in groups.values()]
    return np.concatenate(parts, axis=axis)


if __name__ == "__main__":
    b = load()
    print(f"train {b.X_train.shape}  test {b.X_test.shape}")
    print(f"encoded columns {len(b.columns)}  semantic fields {len(b.groups)}")
    for i, c in enumerate(CLASSES):
        print(f"  {c:7s} train={int((b.y_train == i).sum()):6d} "
              f"test={int((b.y_test == i).sum()):5d}")
