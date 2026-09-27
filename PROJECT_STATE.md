# PROJECT_STATE — updated 2026-09-26 (IST evening)

## Completed stages
- Package `ber` implemented under code/business_entity_resolution (all modules in the master prompt's table; CLI contract implemented).
- 27 unit/contract/leakage/integration tests pass (`python -m pytest src/tests -q`).
- Synthetic multi-country fixture (India/US/France/unknown "Testland", blanks, singletons, 18-vs-81 Oak St, multi same-source matches): all stages run, validator PASS, fixture score 1.0.
- Real-data dev samples carved from train (4000 and 1500 anchors, ~35% distractor rows): audit → split → retrieve → train → freeze(holdout) → infer → validate → score all pass.
- Full-data run: NOT yet executed (needs the 16 GB host; the coding sandbox has 3 GB). See "Next action".

## Current champion
- Sample runs (real data, small target catalog — treat as upper bounds, not full-scale estimates):
  - 4000-anchor sample: retrieval ceiling macro-oracle 0.9995 (link recall 0.9987, 60 cands/anchor); dev macro F0.5 0.9943 (bootstrap CI 0.9926–0.9953), holdout 0.983, pseudo-test 0.9950; all-empty baseline 0.059; exact-rule baseline 0.111.
  - Country-held-out proxies for France: train India → US 0.991; train US → India 0.978 (recall drops to 0.95, precision holds).
  - Error mass (dev): 44% retrieved-but-scored-low, 30% false merge on non-singleton, 15% false merge on singleton, 10% missed by retrieval (mostly Indic-script names; anyascii transliteration view added in normaliser v3).
- Policy: single global calibrated threshold (t≈0.56–0.74 depending on sample; plateau width ≈0.4–0.6 → stable); no one-to-one constraint.

## Next action
1. On the laptop: `pip install -r requirements.txt && pip install -e . --no-deps && python -m pytest src/tests -q`
2. Dry run: `python -m ber.cli fixture --n 4000 --out ..\..\data\sample` then the sample commands in code/business_entity_resolution/README.md.
3. Full run: audit → split → retrieve --split dev → train → freeze → infer → validate → package (README "Full run"). Record real numbers here and in Documentation_template.md.
4. Upload output/matching_results.tsv (manual, portal). Log the SCORED result in SUBMISSIONS.md.

## Known failures
- None open. Optional dense channel untested here (no GPU; disabled).

## Commands
- see code/business_entity_resolution/README.md

## Data hashes
- recorded by `audit` in runs/<run_id>/audit/audit.json (sha256 + bytes per file).
- rows: train S1 2,206,821 / S2 5,034,616 / S3 5,285,603; test S1 1,732,544 / S2 4,887,273 / S3 5,082,316 (France only in test: 259,452 S1).
- train truth: 7,638,365 links (S2 3,693,619; S3 3,944,746); singleton fraction 0.0558; match-count mode 3.

## Runtime / budget
- Deadline 27 Sep 2026 23:59 IST; 5 uploads/day. Full run estimated 5–7 h on i7-1165G7/16 GB (see README). Start by 10:00 IST on 27 Sep at the latest.

## Decision log
- 2026-09-26: restart from the master prompt; previous LightGBM prototype discarded (kept in git history).
- 2026-09-26: normaliser v3 adds deterministic local transliteration (anyascii, ISC) — India retrieval oracle 0.9979 → 0.9988 on the 4k sample; largest remaining retrieval misses were Indic-script names.
- 2026-09-26: union budget 50 / K=20 per channel chosen for full-scale runtime on CPU (sample: 33 cands/anchor at oracle 0.9999); training on 60k dev anchors (~3M pairs) for CatBoost time.
- 2026-09-26: target exclusivity NOT imposed (task allows many-to-one; ablation flag exists).
