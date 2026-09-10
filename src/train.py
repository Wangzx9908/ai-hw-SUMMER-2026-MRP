"""Train the two IDS classifiers that will later be explained.

Two models on purpose:
  * RandomForest  -> exact TreeSHAP is available (fast, deterministic)
  * MLP           -> a small "deep" model, only reachable via model-agnostic
                     KernelSHAP / LIME, which is the realistic case for a
                     production DL-based IDS.

We evaluate on two protocols, because the gap between them is itself part of
the motivation for explainability:
  (a) held-out split of KDDTrain+   -> same attack distribution as training
  (b) the official KDDTest+         -> contains 17 attack types never seen in
                                       training, so accuracy collapses. A
                                       number alone cannot tell an engineer
                                       *why*; attributions can.
"""

from __future__ import annotations

import json
import pathlib
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             accuracy_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import dataset as ds

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEED = 0


def build_models() -> dict[str, object]:
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=None, min_samples_leaf=2,
        class_weight="balanced_subsample", n_jobs=-1, random_state=SEED,
    )
    mlp = Pipeline([
        ("scale", StandardScaler()),
        ("net", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=60,
                              early_stopping=True, random_state=SEED)),
    ])
    return {"random_forest": rf, "mlp": mlp}


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    b = ds.load()

    # (a) internal split, same distribution
    Xtr, Xval, ytr, yval = train_test_split(
        b.X_train, b.y_train, test_size=0.2, stratify=b.y_train, random_state=SEED
    )

    report: dict[str, object] = {
        "n_train": int(b.X_train.shape[0]),
        "n_test_official": int(b.X_test.shape[0]),
        "n_encoded_columns": int(b.X_train.shape[1]),
        "n_semantic_fields": len(b.groups),
        "classes": ds.CLASSES,
        "models": {},
    }

    for name, model in build_models().items():
        t0 = time.perf_counter()
        model.fit(Xtr, ytr)
        fit_s = time.perf_counter() - t0

        pv, pt = model.predict(Xval), model.predict(b.X_test)
        entry = {
            "fit_seconds": round(fit_s, 2),
            "val_accuracy": round(float(accuracy_score(yval, pv)), 4),
            "val_macro_f1": round(float(f1_score(yval, pv, average="macro")), 4),
            "test_accuracy": round(float(accuracy_score(b.y_test, pt)), 4),
            "test_macro_f1": round(float(f1_score(b.y_test, pt, average="macro")), 4),
            "test_per_class_recall": {
                c: round(float(r), 4) for c, r in zip(
                    ds.CLASSES,
                    confusion_matrix(b.y_test, pt, labels=range(5)).diagonal()
                    / np.maximum(np.bincount(b.y_test, minlength=5), 1))
            },
            "test_confusion_matrix": confusion_matrix(
                b.y_test, pt, labels=range(5)).tolist(),
        }
        report["models"][name] = entry
        print(f"\n=== {name} ===")
        print(f"fit {fit_s:.1f}s | val acc {entry['val_accuracy']:.4f} | "
              f"official test acc {entry['test_accuracy']:.4f}")
        print(classification_report(b.y_test, pt, labels=range(5),
                                    target_names=ds.CLASSES, zero_division=0))

        # Refit on the full KDDTrain+ before persisting: the explanation study
        # should analyse the strongest available model.
        model.fit(b.X_train, b.y_train)
        joblib.dump(model, RESULTS / f"{name}.joblib")

    (RESULTS / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {RESULTS / 'metrics.json'}")


if __name__ == "__main__":
    main()
