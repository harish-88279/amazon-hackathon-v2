"""Run directories, statuses and manifests under runs_root/<run_id>/."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from . import io as bio
from .config import Config

STATUSES = ["CREATED", "AUDITED", "SPLIT", "RETRIEVED", "TRAINED", "EVALUATED", "FROZEN", "INFERRED", "VALIDATED", "PACKAGED"]


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"


def run_dir(cfg: Config, run_id: str, create: bool = True) -> Path:
    d = cfg.runs_root / run_id
    if create:
        d.mkdir(parents=True, exist_ok=True)
        if not (d / "manifest.json").exists():
            bio.write_json({"run_id": run_id, "status": "CREATED", "created": dt.datetime.now().isoformat(),
                            "config_hash": cfg.hash(), "history": []}, d / "manifest.json")
            cfg.dump(d / "config.yaml")
    return d


def set_status(cfg: Config, run_id: str, status: str, **extra) -> None:
    assert status in STATUSES, status
    d = run_dir(cfg, run_id)
    m = bio.read_json(d / "manifest.json")
    m["status"] = status
    m["history"].append({"status": status, "at": dt.datetime.now().isoformat(), **extra})
    m.update(extra)
    bio.write_json(m, d / "manifest.json")


def get_manifest(cfg: Config, run_id: str) -> dict:
    return bio.read_json(run_dir(cfg, run_id, create=False) / "manifest.json")


def latest_run(cfg: Config) -> str | None:
    runs = sorted(p.name for p in cfg.runs_root.glob("run_*") if (p / "manifest.json").exists())
    return runs[-1] if runs else None
