# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [TEAM_NAME]  
**Team Members:** Harish HJ [add teammates]  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
A classical retrieve-then-classify entity resolution system: multi-view sparse candidate retrieval (name / address
character TF-IDF, rarity-weighted word TF-IDF, exact keys) per country and target source, a CatBoost pair
classifier over ~90 similarity, missingness and block-context features, cross-fitted probability calibration, and a
decision threshold tuned on the exact macro-F0.5 metric with identity-group validation and an untouched holdout.
No external data, lookups or hosted models are used; the released system has zero neural parameters.

---

## 2. Methodology

### 2.1 Problem Analysis
Measured on the supplied training data (audit.json):
- 2,206,821 S1 / 5,034,616 S2 / 5,285,603 S3 training rows (US 60%, India 40%); test adds France (259,452 of 1,732,544 S1).
- 7,638,365 labelled links; singletons are only 5.6% of S1; the modal match count is 3 (max 11); ~25% of S2/S3 rows are
  unlabelled distractors. An all-empty submission scores ≈0.056; recall therefore matters nearly as much as precision.
- Noise seen in labelled pairs: typos/transpositions, accents, legal-suffix add/drop, domain-style names
  (`maurewilliamscolombier.com`), names written in Indic scripts (Devanagari/Telugu/Tamil), blank addresses,
  abbreviation both ways (`Street`→`ST`→`SAINT`), state names vs codes, dropped house numbers, and completely
  different names at the same address. Hence name-only or address-only retrieval each miss ~5–8% of links; the union
  is required (channel ablation on the 4k sample: name-char 0.928, addr-char 0.948, combined-word 0.986, union 0.9987).
- No target is labelled to more than one S1 in training (label graph audit), but the task statement allows many
  matches per S1 from the same source, so no one-to-one constraint is imposed.

### 2.2 Solution Strategy
**Approach Type:** Blocking + Classifier (retrieve → score → calibrated threshold).  
**Core Innovation:** evidence-preserving multi-view normalisation (incl. deterministic local transliteration),
provenance-rich union retrieval per country/source with block-context features (candidate competition, rank gaps,
name commonness), and decision selection on the exact per-entity macro metric with group-isolated validation.

---

## 3. Candidate Generation (Blocking)
- **Blocking keys used:** (1) char TF-IDF cosine on `name_core` (char_wb 2–5-grams), (2) char TF-IDF on folded
  address (3–5-grams), (3) rarity-weighted word TF-IDF on `name_fold | addr_fold` with field prefixes,
  (4) exact keys: name key (no spaces), sorted name, postal+first name token, house number+street token,
  metaphone+locality token (blocks > 300 ranked by name/address similarity and capped). Same-country routing with a
  global fallback for unknown countries; S2 and S3 indexed separately (K=20 each per channel); union budget 50/anchor.
- **Candidate pairs generated:** [full run: see runs/<run_id>/retrieval_test.json → pairs, pairs_per_anchor, reduction_ratio]
  (4k-anchor sample: 33–60 per anchor, reduction ratio 0.9967).
- **How you ensured true matches were not lost:** measured on labelled dev anchors against the full training target
  catalogue: micro link recall, per-entity recall, fraction of non-singletons fully/partially retrieved and the macro
  oracle (perfect classifier inside candidates) by country/source/match-bucket; per-channel unique positives.
  Sample: link recall 0.9987, macro oracle 0.9995 (US 0.9999, India 0.9988). [full run: retrieval_dev.json]

---

## 4. Matching Model

**Features used:**
- Name features: exact equality on folded/core/sorted/key views; Jaro-Winkler, Levenshtein (core and key), token-set,
  token-sort, partial ratios; token Jaccard/containment; rarity-weighted shared and differing token mass; acronym,
  metaphone and legal-form agreement (4-state); suffix-only difference; length/token-count ratios; word-order-change flag;
  name missingness; non-Latin-script flags.
- Address features: exact/JW/Levenshtein/token-set/sort/Jaccard on folded addresses; postal, house-number and unit
  agreement as present/present-equal, conflict, one-missing, both-missing; ordered numeric-sequence equality and numeric
  Jaccard; street and locality token Jaccard; rarity-weighted shared/differing tokens; parser confidence; missingness.
- Other: retrieval provenance (which channels, channel cosines and ranks, exact-key mask, channel count), source pair,
  country equality/unknown, fields available; block context (block size, rank and gap to best within the anchor for six
  scores, candidate's number of anchors and its best score elsewhere, name commonness); interactions (high name
  similarity with number conflict, high address similarity with low name similarity, hidden number mismatch).

**Model type:** CatBoost binary classifier (depth 8, lr 0.05, ≤1000 iterations with early stopping, L2 3), equal total
weight per anchor, natural retrieved negatives; sigmoid calibration cross-fitted on grouped OOF scores.  
**Threshold selection method:** grid sweep of one global cutoff by exact macro F0.5 on complete S1 prediction sets
(dev OOF), plateau-width check, per-source variant adopted only with a ≥90% bootstrap win rate; confirmed once on the
untouched holdout after freezing.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** [full run: metrics_dev.json → policy.macro_f05 with bootstrap CI; metrics_holdout.json]
  Sample (4k anchors, small catalogue — upper bound): dev 0.9943 (CI 0.9926–0.9953), holdout 0.983, pseudo-test 0.995.
  Baselines: all-empty 0.059; conservative exact name+address rule 0.111; raw score at 0.5 without calibration 0.993.
  Country-held-out proxies: train India→US 0.991; train US→India 0.978.
- **Common false positives (wrong merges):** near-duplicate branches of the same brand at different addresses when the
  address is partial; singletons with a same-name distractor whose address is missing on one side.
- **Common false negatives (missed matches):** heavily corrupted names (multiple transpositions) combined with a
  truncated address; blank-name targets; Indic-script names before the transliteration view (largely fixed).
  Lost-score attribution on the sample: 44% scored-too-low, 30% false merge (non-singleton), 15% singleton false merge,
  10% missed by retrieval.

---

## 6. Conclusion
A carefully validated classical pipeline reaches a high macro F0.5 on identity-group-isolated validation with a stable
threshold plateau and reasonable cross-country transfer, using only the supplied data. The remaining error mass is in
hard near-duplicates; the next measured steps would be hard-negative re-weighting and an entity-level set selector.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` — package `ber` under `src/ber/`, tests under `src/tests/`, `configs/baseline.yaml`
and the frozen `configs/final.yaml`, pinned `requirements.txt`, `pyproject.toml`, `LICENSES/NOTICE.md`,
`release_assets/` (model.cbm, calibration.json, idf.json, model.json, metrics). Entry point `python -m ber.cli`:
`audit → split → retrieve → train → evaluate → error-analysis → freeze → infer → validate → package`.
Reproduce both outputs: run the "Full run" block in `code/business_entity_resolution/README.md`
(inference only: `python -m ber.cli infer --config configs/final.yaml` with the shipped release assets copied to
`runs/<run_id>/`).

### B. Additional Results
- Retrieval channel ablation, calibration reliability tables, threshold curve, per-country/per-bucket slices,
  bootstrap CIs and the top error cases are in `runs/<run_id>/` (`retrieval_dev.json`, `metrics_dev.json`,
  `threshold_curve.json`, `error_report_dev.csv`) and summarised in `PROJECT_STATE.md`.
