"""Build <team>_submission.zip with the exact required layout and a release manifest; rehearse it."""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from . import io as bio
from .config import Config

log = logging.getLogger(__name__)
EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".venv", "venv", ".git", "cache", "runs", "releases", "data", "ber.egg-info", "release_assets"}


def _add_tree(z: zipfile.ZipFile, src: Path, arc_root: str) -> None:
    for p in sorted(src.rglob("*")):
        if any(part in EXCLUDE_DIRS for part in p.relative_to(src).parts):
            continue
        if p.is_file() and not p.name.endswith((".pyc", ".tmp")):
            z.write(p, f"{arc_root}/{p.relative_to(src).as_posix()}")


def build(cfg: Config, team: str, run_id: str, doc_path: Path, summary_path: Path | None = None) -> Path:
    code_dir = Path(__file__).resolve().parents[2]  # code/business_entity_resolution (the package root)
    out_dir = cfg.output_root
    run_dir = cfg.runs_root / run_id
    releases = cfg.releases_root
    for req in (out_dir / "matching_results.tsv", out_dir / "candidate_pairs.tsv", doc_path, code_dir / "README.md", code_dir / "requirements.txt"):
        if not req.exists():
            raise FileNotFoundError(f"required for packaging: {req}")
    zip_path = releases / f"{team}_submission.zip"
    if zip_path.exists():
        zip_path.replace(releases / f"{team}_submission_prev_{zip_path.stat().st_mtime_ns}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out_dir / "matching_results.tsv", "output/matching_results.tsv")
        z.write(out_dir / "candidate_pairs.tsv", "output/candidate_pairs.tsv")
        z.write(doc_path, "Documentation_template.md")
        if summary_path and summary_path.exists():
            z.write(summary_path, "APPROACH_SUMMARY.md")
        _add_tree(z, code_dir, "code/business_entity_resolution")
        # selected model assets so inference is reproducible from saved artefacts
        for name in ("model.json", "calibration.json", "idf.json", "metrics_dev.json", "metrics_holdout.json", "config.yaml", "manifest.json"):
            p = run_dir / name
            if p.exists():
                z.write(p, f"code/business_entity_resolution/release_assets/{name}")
        mf = bio.read_json(run_dir / "model.json")["model_file"]
        z.write(run_dir / mf, f"code/business_entity_resolution/release_assets/{mf}")
    manifest = {"team": team, "run_id": run_id, "zip": str(zip_path), "zip_sha256": bio.sha256_file(zip_path),
                "matching_sha256": bio.sha256_file(out_dir / "matching_results.tsv"), "candidates_sha256": bio.sha256_file(out_dir / "candidate_pairs.tsv"),
                "zip_bytes": zip_path.stat().st_size}
    bio.write_json(manifest, releases / f"{team}_release_manifest.json")
    # immutable copy of the released outputs
    rel_dir = releases / run_id
    rel_dir.mkdir(exist_ok=True)
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        shutil.copy2(out_dir / f, rel_dir / f)
    log.info("packaged %s (%.1f MB)", zip_path, manifest["zip_bytes"] / 1e6)
    return zip_path


def rehearse(zip_path: Path) -> dict:
    """Unzip to a temp dir, check required paths, run the unit tests from inside the package."""
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(td)
            names = set(z.namelist())
        required = ["output/matching_results.tsv", "output/candidate_pairs.tsv", "Documentation_template.md",
                    "code/business_entity_resolution/README.md", "code/business_entity_resolution/requirements.txt",
                    "code/business_entity_resolution/src/ber/cli.py"]
        missing = [r for r in required if r not in names]
        code = Path(td) / "code" / "business_entity_resolution"
        r = subprocess.run([sys.executable, "-m", "pytest", "src/tests", "-q", "-x", "-p", "no:cacheprovider"], cwd=str(code), capture_output=True, text=True,
                           env={**__import__("os").environ, "PYTHONPATH": str(code / "src")})
        return {"missing_paths": missing, "tests_returncode": r.returncode, "tests_tail": (r.stdout + r.stderr)[-1500:], "ok": not missing and r.returncode == 0}
