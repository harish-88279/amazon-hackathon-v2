"""Command line contract.

    python -m ber.cli audit          --config configs/baseline.yaml [--data-dir PATH]
    python -m ber.cli split          --config configs/baseline.yaml [--run-id RUN_ID]
    python -m ber.cli retrieve       --config configs/baseline.yaml --split dev --run-id RUN_ID
    python -m ber.cli train          --config configs/baseline.yaml --run-id RUN_ID [--no-stress]
    python -m ber.cli evaluate       --run-id RUN_ID --split dev
    python -m ber.cli error-analysis --run-id RUN_ID --split dev
    python -m ber.cli freeze         --run-id RUN_ID --output configs/final.yaml
    python -m ber.cli infer          --config configs/final.yaml --split test
    python -m ber.cli validate       --output-dir ../../output [--config ...]
    python -m ber.cli package        --config configs/final.yaml --team TEAM_NAME
    python -m ber.cli fixture        --n 3000 --out ../../data/sample        (dev sample from train)
    python -m ber.cli score-fixture  --config <sample config>                (score a fixture's hidden GT)
Every command exits non-zero on failure and prints a JSON summary on success.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import config as C
from . import io as bio
from . import runs
from .state import update as state_update

log = logging.getLogger("ber")


def _cfg(args, required=True):
    path = getattr(args, "config", None) or "configs/baseline.yaml"
    cfg = C.load(path)
    if getattr(args, "data_dir", None):
        cfg = cfg.with_overrides(data_root=str(Path(args.data_dir).resolve()))
    return cfg


def _run_id(cfg, args, create=False):
    rid = getattr(args, "run_id", None) or cfg.get("frozen.run_id") or runs.latest_run(cfg)
    if not rid:
        if create:
            rid = runs.new_run_id()
        else:
            raise SystemExit("no run id: pass --run-id or create one with `split`")
    runs.run_dir(cfg, rid)
    return rid


def _state(cfg, **kw):
    state_update(cfg.project_root / "PROJECT_STATE.md", **kw)


def _out(obj):
    print(json.dumps(obj, indent=2, default=str))


def cmd_audit(args):
    from .audit import run_audit
    cfg = _cfg(args)
    rid = _run_id(cfg, args, create=True)
    rep = run_audit(cfg, runs.run_dir(cfg, rid) / "audit")
    runs.set_status(cfg, rid, "AUDITED")
    _state(cfg, stages=f"- audit OK (run {rid})", hashes="\n".join(f"- {k}: {v.get('sha256', '')[:16]} ({v.get('bytes')} B)" for k, v in rep["manifest"].items()),
           runtime=json.dumps(rep["environment"]), next=f"python -m ber.cli split --config {args.config} --run-id {rid}")
    _out({"run_id": rid, "environment": rep["environment"], "truth": rep["truth"], "eda_rows": {k: v["rows"] for k, v in rep["eda"].items()}})


def cmd_split(args):
    from .pipeline import stage_split
    cfg = _cfg(args)
    rid = _run_id(cfg, args, create=True)
    man = stage_split(cfg, rid)
    _state(cfg, next=f"python -m ber.cli retrieve --config {args.config} --split dev --run-id {rid}")
    _out({"run_id": rid, "hash": man["hash"], "summary": man["summary"], "multi_s1_groups": man["multi_s1_groups"]})


def cmd_retrieve(args):
    from .pipeline import stage_retrieve
    cfg = _cfg(args)
    rid = _run_id(cfg, args)
    stats = stage_retrieve(cfg, rid, args.split, force=args.force)
    if args.split == "dev":
        _state(cfg, next=f"python -m ber.cli train --config {args.config} --run-id {rid}")
    _out({"run_id": rid, **stats})


def cmd_train(args):
    from .pipeline import stage_train
    cfg = _cfg(args)
    rid = _run_id(cfg, args)
    m = stage_train(cfg, rid, stress=not args.no_stress)
    _state(cfg, champion=f"run {rid}: dev macro F0.5 {m['policy']['macro_f05']:.4f} policy {m['policy'].get('policy')} t_s2={m['policy']['t_s2']} t_s3={m['policy']['t_s3']}",
           next=f"python -m ber.cli freeze --config {args.config} --run-id {rid} --output configs/final.yaml",
           log=f"- {rid}: trained; dev {m['policy']['macro_f05']:.4f}; baselines {m['baselines']}")
    _out({"run_id": rid, "policy": m["policy"], "overall": m["slices"]["overall"], "by_country": m["slices"].get("country"),
          "bootstrap": m["bootstrap"], "baselines": m["baselines"], "stress": m.get("stress"), "error_summary": m["error_summary"]})


def cmd_evaluate(args):
    cfg = _cfg(args)
    rid = _run_id(cfg, args)
    p = runs.run_dir(cfg, rid, create=False) / f"metrics_{args.split}.json"
    if not p.exists():
        raise SystemExit(f"no metrics for split {args.split} in run {rid} (run train / freeze first)")
    _out(bio.read_json(p))


def cmd_error_analysis(args):
    cfg = _cfg(args)
    rid = _run_id(cfg, args)
    p = runs.run_dir(cfg, rid, create=False) / f"error_report_{args.split}.csv"
    if not p.exists():
        raise SystemExit(f"no error report for {args.split} in run {rid}")
    import pandas as pd
    df = pd.read_csv(p)
    m = bio.read_json(runs.run_dir(cfg, rid, create=False) / f"metrics_{args.split}.json")
    _out({"summary": m.get("error_summary"), "top": df.head(args.top).to_dict(orient="records"), "path": str(p)})


def cmd_freeze(args):
    from .pipeline import stage_freeze
    cfg = _cfg(args)
    rid = _run_id(cfg, args)
    hm = stage_freeze(cfg, rid, Path(args.output))
    _state(cfg, next=f"python -m ber.cli infer --config {args.output} --split test",
           log=f"- {rid}: frozen -> {args.output}; holdout macro F0.5 {hm['slices']['overall']['macro_f05']:.4f}")
    _out({"run_id": rid, "final_config": args.output, "holdout": hm["slices"]["overall"], "by_country": hm["slices"].get("country"), "bootstrap": hm["bootstrap"]})


def cmd_infer(args):
    from .pipeline import stage_infer
    cfg = _cfg(args)
    rid = _run_id(cfg, args)
    rep = stage_infer(cfg, rid, args.split, force=args.force)
    ok = rep["hard_checks"]["ok"] and rep["official_validator"].get("pass")
    _state(cfg, next=(f"python -m ber.cli package --config {args.config} --team TEAM_NAME" if ok else "fix validation problems, then rerun infer"),
           log=f"- {rid}: inferred {rep['n_s1']} S1, {rep['n_matches']} matches, validator {'PASS' if ok else 'FAIL'}; sha256 {rep['hard_checks']['sha256_matching'][:16]}")
    _out(rep)
    if not ok:
        sys.exit(1)


def cmd_validate(args):
    from .validate import hard_checks, official
    cfg = _cfg(args)
    out_dir = Path(args.output_dir).resolve()
    rep = {"hard_checks": hard_checks(out_dir, cfg.data_root / "test"), "official_validator": official(cfg.validator, out_dir, cfg.data_root / "test")}
    _out(rep)
    if not (rep["hard_checks"]["ok"] and rep["official_validator"].get("pass")):
        sys.exit(1)


def cmd_package(args):
    from .package import build, rehearse
    cfg = _cfg(args)
    rid = _run_id(cfg, args)
    doc = cfg.project_root / "Documentation_template.md"
    summ = cfg.project_root / "APPROACH_SUMMARY.md"
    zp = build(cfg, args.team, rid, doc, summ)
    reh = rehearse(zp)
    runs.set_status(cfg, rid, "PACKAGED", zip=str(zp))
    _state(cfg, stages=f"- packaged {zp.name}", next="upload output/matching_results.tsv on the portal (manual)", log=f"- {rid}: packaged {zp} rehearsal ok={reh['ok']}")
    _out({"zip": str(zp), "sha256": bio.sha256_file(zp), "rehearsal": reh})
    if not reh["ok"]:
        sys.exit(1)


def cmd_fixture(args):
    from .fixture import from_train, synthetic
    out = Path(args.out).resolve()
    if args.synthetic:
        d = synthetic(out)
    else:
        cfg = _cfg(args)
        d = from_train(cfg.data_root, out, args.n, cfg.seed, cfg.validator)
    import yaml
    base = yaml.safe_load(open(Path(args.config) if getattr(args, "config", None) else "configs/baseline.yaml", encoding="utf-8"))
    base.update(project_root=str(out), data_root=str(out / "dataset"), output_root=str(out / "output"), runs_root=str(out / "runs"),
                cache_root=str(out / "cache"), releases_root=str(out / "releases"), validator=str(out / "utils" / "validate_submission.py"))
    base["training"]["train_s1_sample"] = 0
    base["training"]["catboost"]["iterations"] = 300 if not args.synthetic else 30
    base["decision"]["bootstrap_samples"] = 50
    if args.synthetic:
        base["split"].update(holdout_fraction=0.0, n_folds=2); base["retrieval"]["exact_block_cap"] = 5
    (out / "config.yaml").write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    _out({"dataset": str(d), "config": str(out / "config.yaml"), "next": f"python -m ber.cli audit --config {out / 'config.yaml'}"})


def cmd_score_fixture(args):
    from .metric import summary
    cfg = _cfg(args)
    truth = bio.truth_sets(bio.read_truth(cfg.data_root / "test" / "hidden_ground_truth.tsv"))
    pred = bio.read_id_list_tsv(cfg.output_root / "matching_results.tsv", "matched_entity_ids")
    ids = list(truth)
    _out(summary(truth, {k: set(v) for k, v in pred.items()}, ids))


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="ber", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    def add(name, fn, **opts):
        p = sub.add_parser(name)
        p.add_argument("--config", default="configs/baseline.yaml")
        p.add_argument("--run-id")
        for k, v in opts.items():
            p.add_argument(k, **v)
        p.set_defaults(fn=fn)
        return p
    add("audit", cmd_audit, **{"--data-dir": {}})
    add("split", cmd_split)
    add("retrieve", cmd_retrieve, **{"--split": {"default": "dev", "choices": ["dev", "holdout", "test"]}, "--force": {"action": "store_true"}})
    add("train", cmd_train, **{"--no-stress": {"action": "store_true"}})
    add("evaluate", cmd_evaluate, **{"--split": {"default": "dev", "choices": ["dev", "holdout"]}})
    add("error-analysis", cmd_error_analysis, **{"--split": {"default": "dev"}, "--top": {"type": int, "default": 20}})
    add("freeze", cmd_freeze, **{"--output": {"default": "configs/final.yaml"}})
    add("infer", cmd_infer, **{"--split": {"default": "test", "choices": ["test"]}, "--force": {"action": "store_true"}})
    add("validate", cmd_validate, **{"--output-dir": {"default": "../../output"}})
    add("package", cmd_package, **{"--team": {"required": True}})
    add("fixture", cmd_fixture, **{"--n": {"type": int, "default": 3000}, "--out": {"required": True}, "--synthetic": {"action": "store_true"}})
    add("score-fixture", cmd_score_fixture)
    args = ap.parse_args(argv)
    try:
        args.fn(args)
    except (C.ConfigError, bio.C.ContractError, FileNotFoundError) as e:
        log.error("%s: %s", type(e).__name__, e)
        sys.exit(2)


if __name__ == "__main__":
    main()
