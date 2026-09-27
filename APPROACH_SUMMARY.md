# Amazon ML Challenge 2026 — Business Entity Resolution: approach summary (1–2 pages)

**Task.** For each Source-1 business record, return the Source-2/3 records that are the same real-world
business (zero, one or many). Metric: macro F0.5 per S1 entity, singletons score 1 only for an empty list.
Test adds France, unseen in training. No external data or lookups are allowed.

**Method (classical retrieve-then-classify).**
1. *Normalisation with evidence preservation* — raw, conservative (NFKC, casefold, punctuation split, digits
   kept, `12/3` ≠ `123`) and folded views (accent folding; non-Latin scripts transliterated locally with
   anyascii). Legal-form tokens are separated (`name_core`), never the only representation; address
   components (postal, house number, unit, numeric sequences, street/locality tokens) are extracted with
   confidence flags; `St` inside a name is never expanded.
2. *Multi-view candidate retrieval* per country (unknown countries use a global index) and per target
   source: char TF-IDF on names (2–5-grams), char TF-IDF on addresses (3–5-grams), rarity-weighted word
   TF-IDF on name+address with field markers, exact-key blocks (name key, sorted name, postal+token,
   house+street, metaphone+locality; big blocks ranked and capped). Union with provenance, budget 50/anchor.
   Ceiling on a 4k-anchor real-data sample: link recall 0.9987, macro oracle 0.9995.
3. *Pair classifier* — CatBoost (Apache-2.0) on ~90 features: name/address similarities on several views,
   4-state present/missing agreement codes for postal/house/unit, rarity-weighted shared/differing tokens,
   retrieval provenance and ranks, block context (rank/gap within the anchor's block, candidate competition
   across anchors, name commonness), and interaction flags (high name similarity with number conflicts, etc.).
   Training pairs are the natural retrieved candidates of DEV anchors only; identity-group splits (label-graph
   components) with an untouched 15% holdout; each anchor gets equal total weight.
4. *Calibration + decision* — sigmoid calibration cross-fitted on grouped out-of-fold scores; one global
   threshold selected by exact macro F0.5 on complete prediction sets (per-source thresholds adopted only if a
   bootstrap shows a stable gain); no one-to-one constraint.
5. *Outputs* — candidate_pairs.tsv is exactly the scored pair set (fingerprint-checked); matching_results.tsv
   is a subset; strict format checks plus the official validator.

**Validation.** Grouped 5-fold OOF on dev, bootstrap CIs by identity group, untouched holdout evaluated once
after freezing, country-held-out proxies for France (train India→US 0.991, train US→India 0.978 on the sample).
Sample dev macro F0.5 0.994 (CI 0.993–0.995), holdout 0.983, pseudo-test 0.995; all-empty baseline 0.06.
*Full-data numbers are filled in after the final run (see Documentation_template.md §5).*

**Error analysis (sample).** 44% of lost score: retrieved but scored below threshold (mostly heavy typo /
transliteration variants); 30% false merges among near-duplicate branches; 15% false merges on singletons;
10% missed by retrieval. Transliteration view was the largest single retrieval fix.

**Compliance.** Only supplied data; CatBoost/LightGBM/scikit-learn/rapidfuzz/anyascii (Apache/MIT/BSD/ISC);
0 neural parameters in the released system; optional multilingual-e5-small (MIT, 118M) channel is disabled.
