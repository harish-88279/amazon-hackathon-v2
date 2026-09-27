"""PROJECT_STATE.md maintenance (checkpoint for context resets)."""
from __future__ import annotations

import datetime as dt
from pathlib import Path


def update(path: Path, **sections: str) -> None:
    """Rewrite PROJECT_STATE.md sections (keeps others). Sections: stages, champion, next, failures, commands, hashes, runtime, log."""
    order = ["stages", "champion", "next", "failures", "commands", "hashes", "runtime", "log"]
    titles = {"stages": "Completed stages", "champion": "Current champion", "next": "Next action", "failures": "Known failures",
              "commands": "Commands", "hashes": "Data hashes", "runtime": "Runtime / budget", "log": "Decision log"}
    existing: dict[str, str] = {}
    if path.exists():
        cur, buf = None, []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                if cur: existing[cur] = "\n".join(buf).strip()
                key = next((k for k, t in titles.items() if line[3:].strip() == t), None)
                cur, buf = key, []
            elif cur:
                buf.append(line)
        if cur: existing[cur] = "\n".join(buf).strip()
    if "log" in sections and "log" in existing:
        sections["log"] = existing["log"] + "\n" + sections["log"]
    existing.update(sections)
    out = [f"# PROJECT_STATE — updated {dt.datetime.now().isoformat(timespec='seconds')}", ""]
    for k in order:
        if k in existing:
            out += [f"## {titles[k]}", existing[k], ""]
    path.write_text("\n".join(out), encoding="utf-8")
