# Explainable AI for Network Intrusion Detection

Mini Research Problem — *Artificial Intelligence / Advanced Topics in AI & ML*
(based on Lecture 09: Interpretability & Explainability).

📄 **Presentation: [`slides/mrp.pdf`](slides/mrp.pdf)** — 11 slides, ~10 min

## What this does

An ML-based intrusion detection system only outputs *normal* or *attack*, which
a network engineer cannot act on. This project trains a Random Forest on the
NSL-KDD flow dataset and uses **SHAP** to show which of the 41 flow features
drove each decision.

**Result:** the model relies on exactly the indicators a network engineer would
check manually.

| Attack type | Top SHAP features | Interpretation |
|---|---|---|
| DoS | `src_bytes`=0, `count`, `dst_host_serror_rate`=1 | SYN flood: empty packets, many connections, failed handshakes |
| Probe | `src_bytes`=0, `dst_host_diff_srv_rate`, `logged_in`=0 | Port scan: empty connections spread over many services |

Accuracy on the official test split: **74.3 %** (DoS recall 77 %, Probe 65 %).

## Run it

```bash
./run_all.sh        # downloads NSL-KDD, trains, produces all figures (~1 min)
```

or directly:

```bash
python3 src/nids_xai.py
```

Needs `numpy pandas scikit-learn matplotlib shap`.

## Files

```
src/nids_xai.py    everything: load data -> train Random Forest -> SHAP -> figures
figures/           shap_global.png, shap_local_dos.png, shap_local_probe.png
results/           results.json (accuracy, per-class recall, top features)
slides/mrp.tex     beamer source for the presentation
slides/mrp.pdf     the presentation
```

## Data and tools

- NSL-KDD — <https://www.unb.ca/cic/datasets/nsl.html> (Tavallaee et al., CISDA 2009)
- SHAP — <https://github.com/shap/shap>
- LIME (discussed in the literature review) — <https://github.com/marcotcr/lime>
- scikit-learn — <https://scikit-learn.org/>

## References

- Lundberg & Lee, *A Unified Approach to Interpreting Model Predictions*, NeurIPS 2017.
- Lundberg et al., *From local explanations to global understanding with explainable AI for trees*, Nature Machine Intelligence 2020.
- Ribeiro et al., *"Why Should I Trust You?"*, KDD 2016.
- Sommer & Paxson, *Outside the Closed World*, IEEE S&P 2010.
- Warnecke et al., *Evaluating Explanation Methods for Deep Learning in Security*, IEEE EuroS&P 2020.
- Wang et al., *An Explainable Machine Learning Framework for Intrusion Detection Systems*, IEEE Access 2020.
# Explainable AI for Network Intrusion Detection

Mini Research Problem — *Artificial Intelligence / Advanced Topics in AI & ML*,
built on Lecture 09 (Interpretability, Explainability, AI Ethics).

**Question.** An ML-based IDS outputs *block* or *allow*. Before a network
engineer lets it touch production traffic, they need the reason. So: are SHAP
explanations of a flow classifier **faithful**, **stable**, and **consistent
with network protocol semantics** — and cheap enough to run?

📄 **Presentation: [`slides/mrp_xai_nids.pdf`](slides/mrp_xai_nids.pdf)** (14 slides, ~10 min)

## Headline results

| | |
|---|---|
| Random Forest, same-distribution held-out split | **99.9 %** accuracy |
| Random Forest, official `KDDTest+` (17 novel attacks) | **74.8 %** accuracy |
| Recall per family on unseen traffic | DoS 77 % · Probe 60 % · R2L 7 % · U2R 7 % |
| Faithfulness, AOPC (higher = better) | **TreeSHAP 0.69** · LIME 0.44 · random 0.22 |
| Stability across identical re-runs, top-5 Jaccard | **TreeSHAP 1.00** · LIME 0.66 |
| Cost per explained flow | TreeSHAP 47 ms · LIME 74 ms · KernelSHAP/MLP 31 ms |
| corr(domain alignment, recall on unseen attacks) | **0.90** over the 4 attack families |

Three findings worth the slide space:

1. **The learned logic reproduces textbook signatures.** DoS attributions
   concentrate on `count`, `flag=REJ`, `serror_rate`; Probe on `src_bytes`≈0 with
   high `diff_srv_rate`; R2L on `logged_in`, `hot`, `is_guest_login`.
2. **Domain alignment predicts generalisation.** The share of attribution mass
   landing in the feature groups an analyst would use (fixed *before* looking at
   any output) gives a lift of 1.45 for DoS down to 0.83 for U2R — and that lift
   correlates 0.90 with per-family recall on unseen attacks. The explanation
   exposed *why* R2L/U2R fail without collecting new labels.
3. **SHAP and LIME largely disagree** on the same decision (top-5 Jaccard 0.41,
   Spearman −0.12). The deletion test breaks the tie in SHAP's favour, matching
   Warnecke et al. (EuroS&P 2020).

## Reproduce

```bash
./run_all.sh          # ~6 min on a laptop CPU, downloads NSL-KDD itself
```

Requires Python ≥ 3.10 and, for the slides, a TeX distribution with `beamer`.

## Layout

```
src/dataset.py    NSL-KDD loading, one-hot encoding + the semantic group map
src/train.py      Random Forest and MLP, evaluated on both protocols
src/explain.py    TreeSHAP global/local figures, firewall-style reason strings
src/xai_eval.py   E1 domain consistency, E2 faithfulness, E3 stability, E4 cost
src/report.py     final figure + slides/numbers.tex (every number on a slide)
slides/           beamer source and the compiled PDF
results/          metrics.json, xai_eval.json, shap_global.json, reason_strings.txt
```

Every figure and every number quoted in the deck is generated by this code:
`report.py` writes `slides/numbers.tex`, and the `.tex` only references those
macros, so the slides cannot drift away from the results.

## One design decision worth noting

One-hot encoding expands the 41 NSL-KDD protocol fields into 122 columns, and a
per-column attribution (`service=ecr_i`: 0.004) is useless in an incident
report. Shapley values are additive, so summing over a disjoint group of columns
is *exactly* the attribution of the original field. All results are reported on
the 41 semantic fields — see `aggregate_to_fields` in `src/dataset.py`.

## Data and tools

- NSL-KDD — <https://www.unb.ca/cic/datasets/nsl.html> (Tavallaee et al., CISDA 2009)
- CICIDS2017, natural next step — <https://www.unb.ca/cic/datasets/ids-2017.html>
- `shap` — <https://github.com/shap/shap> · `lime` — <https://github.com/marcotcr/lime>
- `scikit-learn` — <https://scikit-learn.org/>

## Key references

- Lundberg & Lee, *A Unified Approach to Interpreting Model Predictions*, NeurIPS 2017.
- Lundberg et al., *From local explanations to global understanding with explainable AI for trees*, Nature Machine Intelligence 2020.
- Ribeiro et al., *"Why Should I Trust You?" Explaining the Predictions of Any Classifier*, KDD 2016.
- Sommer & Paxson, *Outside the Closed World: On Using Machine Learning for Network Intrusion Detection*, IEEE S&P 2010.
- Arp et al., *Dos and Don'ts of Machine Learning in Computer Security*, USENIX Security 2022.
- Warnecke et al., *Evaluating Explanation Methods for Deep Learning in Security*, IEEE EuroS&P 2020.
- Wang et al., *An Explainable Machine Learning Framework for Intrusion Detection Systems*, IEEE Access 2020.
- Nadeem et al., *SoK: Explainable Machine Learning for Computer Security Applications*, IEEE EuroS&P 2023.

## Limitations

NSL-KDD features describe 1999 traffic. The deletion test uses a benign-median
reference that sits close to TreeSHAP's implicit background, a mild bias in
SHAP's favour. All metrics are functionally grounded — no operator study.
