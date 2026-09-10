#!/usr/bin/env bash
# Reproduce every number and figure in slides/mrp_xai_nids.pdf from scratch.
#
#   ./run_all.sh
#
# Runtime on a laptop CPU: ~6 min (LIME in xai_eval.py dominates).
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python3}

echo "== 0/5 dependencies"
$PY -m pip install --quiet --user numpy pandas scikit-learn matplotlib shap lime

echo "== 1/5 dataset (NSL-KDD, official splits)"
mkdir -p data figures results
for f in KDDTrain+.txt KDDTest+.txt; do
  [ -s "data/$f" ] || curl -sSL -o "data/$f" \
    "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/$f"
done
(cd src && $PY dataset.py)

echo "== 2/5 train Random Forest + MLP"
(cd src && $PY train.py)

echo "== 3/5 SHAP global and local explanations"
(cd src && $PY explain.py)

echo "== 4/5 explanation evaluation E1-E4"
(cd src && $PY xai_eval.py)

echo "== 5/5 slide numbers and final figure"
(cd src && $PY report.py)

if command -v pdflatex >/dev/null; then
  echo "== slides"
  (cd slides && pdflatex -interaction=nonstopmode mrp_xai_nids.tex >/dev/null \
    && pdflatex -interaction=nonstopmode mrp_xai_nids.tex >/dev/null)
  echo "   -> slides/mrp_xai_nids.pdf"
else
  echo "pdflatex not found, skipping the slide build"
fi
echo "done"
