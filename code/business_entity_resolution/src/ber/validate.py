"""Hard output checks (stricter than the organizer script) + the official validator."""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

from . import contracts as C
from . import io as bio

log = logging.getLogger(__name__)


def hard_checks(output_dir: Path, test_dir: Path, scored_fingerprint: tuple[int, int] | None = None) -> dict:
    m_path, c_path = output_dir / "matching_results.tsv", output_dir / "candidate_pairs.tsv"
    matches = bio.read_id_list_tsv(m_path, "matched_entity_ids")
    cands = bio.read_id_list_tsv(c_path, "candidate_entity_ids")
    s1 = bio.read_tsv(test_dir / "test_source1.tsv", C.SOURCE_COLUMNS)
    s1_ids = s1["entity_id"].tolist()
    tgt = set()
    for f in ("test_source2.tsv", "test_source3.tsv"):
        tgt |= set(bio.read_tsv(test_dir / f, C.SOURCE_COLUMNS)["entity_id"])
    problems = []
    if list(matches) != s1_ids:
        problems.append("matching_results rows are not exactly the test S1 ids in input order")
    if set(cands) != set(s1_ids):
        problems.append("candidate_pairs S1 id set differs from test S1")
    n_bad_tgt = n_s1_in_list = n_subset = n_pairs = 0
    for a in s1_ids:
        mm, cc = set(matches.get(a, [])), set(cands.get(a, []))
        n_pairs += len(cc)
        for x in mm | cc:
            if x.startswith("S1-"):
                n_s1_in_list += 1
            elif x not in tgt:
                n_bad_tgt += 1
        if not mm <= cc:
            n_subset += 1
    if n_bad_tgt: problems.append(f"{n_bad_tgt} list ids do not exist in test S2/S3")
    if n_s1_in_list: problems.append(f"{n_s1_in_list} S1 ids inside lists")
    if n_subset: problems.append(f"{n_subset} anchors whose matches are not a subset of candidates")
    if scored_fingerprint is not None:
        from .pipeline import _pair_fingerprint
        emitted = _pair_fingerprint((a, x) for a, xs in cands.items() for x in xs)
        if emitted != tuple(scored_fingerprint):
            problems.append(f"candidate file pairs != scored pair set (emitted {emitted[0]} pairs, scored {scored_fingerprint[0]})")
    # round trip of empties: re-read raw text
    with open(m_path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n")
        if header != "source1_entity_id\tmatched_entity_ids":
            problems.append(f"bad header {header!r}")
        for ln, line in enumerate(f, 2):
            if line.count("\t") != 1:
                problems.append(f"line {ln} does not have exactly one tab"); break
    counts = s1["country"].value_counts().to_dict()
    rep = {"ok": not problems, "problems": problems, "n_s1": len(s1_ids), "country_counts": counts,
           "n_candidate_pairs": n_pairs, "n_matches": sum(len(v) for v in matches.values()),
           "n_empty_matches": sum(1 for v in matches.values() if not v), "n_empty_candidates": sum(1 for v in cands.values() if not v),
           "sha256_matching": bio.sha256_file(m_path), "sha256_candidates": bio.sha256_file(c_path)}
    return rep


def official(validator: Path, output_dir: Path, test_dir: Path) -> dict:
    if not validator.exists():
        return {"available": False, "note": "official validator missing - official validation PENDING"}
    cmd = [sys.executable, str(validator), "--matching", str(output_dir / "matching_results.tsv"),
           "--candidate", str(output_dir / "candidate_pairs.tsv"), "--test-dir", str(test_dir)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(validator.parent.parent))
    return {"available": True, "cmd": " ".join(cmd), "returncode": r.returncode, "stdout": r.stdout[-4000:], "stderr": r.stderr[-2000:],
            "pass": r.returncode == 0 and "PASS" in r.stdout}
