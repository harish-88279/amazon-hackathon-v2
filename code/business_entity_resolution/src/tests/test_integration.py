"""End-to-end run of every CLI stage on the synthetic multi-country fixture, in a temp dir."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _run(args, cwd):
    r = subprocess.run([sys.executable, "-m", "ber.cli", *args], cwd=str(cwd), capture_output=True, text=True,
                       env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    return r.stdout


@pytest.mark.slow
def test_cli_end_to_end(tmp_path):
    out = tmp_path / "fx"
    _run(["fixture", "--synthetic", "--out", str(out)], ROOT)
    (out / "utils").mkdir()
    (out / "utils" / "validate_submission.py").write_text("import sys\nprint('PASS (stub validator)')\n", encoding="utf-8")
    cfg = yaml.safe_load((ROOT / "configs" / "baseline.yaml").read_text())
    cfg.update(project_root=str(out), data_root=str(out / "dataset"), output_root=str(out / "output"), runs_root=str(out / "runs"),
               cache_root=str(out / "cache"), releases_root=str(out / "releases"), validator=str(out / "utils" / "validate_submission.py"))
    cfg["split"].update(holdout_fraction=0.0, n_folds=2)
    cfg["training"].update(train_s1_sample=0)
    cfg["training"]["catboost"]["iterations"] = 30
    cfg["decision"]["bootstrap_samples"] = 5
    cfg["retrieval"]["exact_block_cap"] = 5
    c = out / "cfg.yaml"
    c.write_text(yaml.safe_dump(cfg))
    _run(["audit", "--config", str(c)], ROOT)
    rid = sorted(p.name for p in (out / "runs").iterdir())[-1]
    _run(["split", "--config", str(c), "--run-id", rid], ROOT)
    _run(["retrieve", "--config", str(c), "--split", "dev", "--run-id", rid], ROOT)
    _run(["train", "--config", str(c), "--run-id", rid, "--no-stress"], ROOT)
    final = out / "final.yaml"
    _run(["freeze", "--config", str(c), "--run-id", rid, "--output", str(final)], ROOT)
    _run(["infer", "--config", str(final)], ROOT)
    _run(["validate", "--config", str(final), "--output-dir", str(out / "output")], ROOT)
    m = (out / "output" / "matching_results.tsv").read_text(encoding="utf-8").splitlines()
    assert m[0] == "source1_entity_id\tmatched_entity_ids" and len(m) == 9
    rows = dict(l.split("\t") for l in m[1:])
    assert rows["S1-7"] == "" and "S1-7" in rows            # unknown country + blank fields -> one valid (empty) row
    assert rows["S1-6"] == ""                               # true singleton stays empty
    assert "S3-2" not in rows["S1-3"] and "S2-3" not in rows["S1-4"]   # 18 vs 81 Oak Street not merged
    assert set(rows["S1-8"].split(",")) >= {"S2-6", "S2-7"}  # multiple same-source matches retained
    s = json.loads(_run(["score-fixture", "--config", str(final)], ROOT))
    assert s["macro_f05"] >= 0.9
