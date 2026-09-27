# Amazon ML Challenge 2026 — Business Entity Resolution

Team repository. Start with **`PROJECT_STATE.md`** (status, champion, next action, decision log), then
**`code/business_entity_resolution/README.md`** (environment, dry run, full run, packaging).

- `code/business_entity_resolution/` — the `ber` package, configs, tests, README, requirements
- `student_resource/` — organizer files (README, template, validator); **`student_resource/dataset/` is not in git**
  (download `student_resource.zip` from the Unstop round page and extract so the TSVs land in
  `student_resource/dataset/{train,test}/`)
- `Documentation_template.md` — methodology write-up (goes in the zip root); `APPROACH_SUMMARY.md` — 1–2 page summary
- `SUBMISSIONS.md` — every leaderboard upload
- `runs/`, `cache/`, `output/`, `releases/`, `data/` — generated, not in git

Rules: no external data/APIs/geocoding; model MIT/Apache and ≤ 8B params (ours: CatBoost, 0 neural params).
