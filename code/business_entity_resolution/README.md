# Business Entity Resolution — Amazon ML Challenge 2026 (`ber`)

Classical retrieve-then-classify pipeline: multi-view sparse candidate retrieval → CatBoost pair
classifier with block-context features → cross-fitted probability calibration → decision policy
tuned on the exact macro-F0.5 metric → strict output validation. No external data, APIs,
geocoders or hosted models are used. Model licence: CatBoost (Apache-2.0); trained artefacts
are ours. Neural parameter count of the released system: 0.

Full write-up: `../../Documentation_template.md`. Short summary: `../../APPROACH_SUMMARY.md`.
Project state / decision log: `../../PROJECT_STATE.md`.

## Layout

```
configs/baseline.yaml     all knobs (paths, retrieval K/budgets, model, calibration, decision policy)
configs/final.yaml        frozen configuration written by `freeze` (run id + decision policy)
src/ber/
  contracts.py io.py      strict TSV parsing, id rules, exact output serialisation, manifests
  metric.py               exact per-entity + macro F0.5 (two independent implementations)
  audit.py                data audit: hashes, integrity, label graph, EDA
  splits.py               identity-group components, untouched holdout + grouped folds, leakage asserts
  normalize.py parse_address.py   raw / conservative / folded(+transliterated) views, address components
  data.py                 content-addressed cache of normalised tables (Arrow-backed, country filters)
  retrieve.py             channels: name char-TFIDF, address char-TFIDF, combined word-TFIDF, exact keys,
                          (optional dense E5); union + budget; ceiling / oracle statistics
  features.py             pair + context features, versioned schema, chunked to Parquet row groups
  train.py calibrate.py   CatBoost/LightGBM fits, grouped OOF, sigmoid/isotonic cross-fit calibration
  decide.py               threshold sweep by macro F0.5, per-source option, bootstrap CI
  evaluate.py error_analysis.py   slices, baselines, cause-attributed error report
  pipeline.py             stage orchestration; streaming inference
  validate.py package.py  hard output checks + official validator; release zip + rehearsal
  cli.py runs.py state.py fixture.py   command line, run manifests, PROJECT_STATE.md, dev fixtures
src/tests/                metric, contracts, normalisation, splits/leakage, retrieval/features, CLI integration
```

## Environment

Python 3.10 or 3.11, 16 GB RAM, CPU only. Windows PowerShell:
```powershell
cd "D:\Amazon Hackathon\code\business_entity_resolution"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pytest src/tests -q
```
(bash: `source .venv/bin/activate`.) The dataset must be at `../../student_resource/dataset/{train,test}`
(configured in `configs/baseline.yaml` as `data_root`; override per command with `--data-dir`).

## 10-minute dry run on a real-data sample (do this first)

```powershell
python -m ber.cli fixture --n 4000 --out ..\..\data\sample
python -m ber.cli audit    --config ..\..\data\sample\config.yaml
python -m ber.cli split    --config ..\..\data\sample\config.yaml
python -m ber.cli retrieve --config ..\..\data\sample\config.yaml --split dev
python -m ber.cli train    --config ..\..\data\sample\config.yaml
python -m ber.cli freeze   --config ..\..\data\sample\config.yaml --output ..\..\data\sample\final.yaml
python -m ber.cli infer    --config ..\..\data\sample\final.yaml
python -m ber.cli score-fixture --config ..\..\data\sample\final.yaml
```
(`--run-id` defaults to the latest run under the config's `runs_root`.)

## Full run (official data)

```powershell
python -m ber.cli audit          --config configs/baseline.yaml            # ~10 min (normalises all 7 files once)
python -m ber.cli split          --config configs/baseline.yaml
python -m ber.cli retrieve       --config configs/baseline.yaml --split dev # 60k dev anchors vs full train targets
python -m ber.cli train          --config configs/baseline.yaml            # features, 5-fold OOF, calibration, sweep, stress tests
python -m ber.cli evaluate       --run-id RUN_ID --split dev
python -m ber.cli error-analysis --run-id RUN_ID --split dev
python -m ber.cli freeze         --run-id RUN_ID --output configs/final.yaml   # + one untouched holdout evaluation
python -m ber.cli infer          --config configs/final.yaml --split test      # 1.73M anchors -> output/*.tsv + validators
python -m ber.cli validate       --output-dir ../../output --config configs/final.yaml
python -m ber.cli package        --config configs/final.yaml --team TEAM_NAME
```
Expected wall time on an i7-1165G7 / 16 GB (rough, measured on samples and extrapolated):
audit ~10 min · dev retrieval ~30 min · train ~1–1.5 h · freeze ~15 min · test retrieval ~1–2 h ·
test features + scoring ~2 h. Start early; run `audit` → `train` first, then `freeze` → `infer`.
If memory is tight lower `inference.pair_chunk` (1.5M → 750k) and `training.train_s1_sample`.

Outputs: `../../output/matching_results.tsv` (upload), `../../output/candidate_pairs.tsv` (package).
Artefacts per run under `../../runs/<run_id>/`: `audit/audit.json`, `splits/`, `retrieval_*.json`
(ceilings), `metrics_dev.json`, `metrics_holdout.json`, `error_report_dev.csv`, `model.cbm`,
`model.json`, `calibration.json`, `threshold_curve.json`, `inference_test.json` (validator output).
Releases: `../../releases/<team>_submission.zip` + manifest with SHA256s.

## Notes

* Country is an open set: known countries query the same-country index, unknown/missing ones the
  global index; every anchor always gets a row (empty if no candidates).
* Target exclusivity (one S1 per S2/S3) is NOT assumed; `decision.one_to_one` exists only as a
  labelled-split ablation.
* Optional dense channel (`retrieval.dense.enabled: true`, intfloat/multilingual-e5-small, MIT) needs
  `sentence-transformers` + `faiss-cpu`; it is off — CPU-only inference on 10M rows did not fit the budget.
