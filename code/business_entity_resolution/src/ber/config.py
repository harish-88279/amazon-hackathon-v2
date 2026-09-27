"""Configuration loading and validation.

A config is a YAML file (see configs/baseline.yaml). Relative paths are resolved against the
folder that contains the config file's parent ``configs/`` directory, i.e. the package root
``code/business_entity_resolution/``. Values are validated before any long-running work.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


@dataclass
class Config:
    raw: dict[str, Any]
    base_dir: Path                      # folder the config paths are relative to
    path: Path | None = None
    resolved: dict[str, Path] = field(default_factory=dict)

    # ------------------------------------------------------------ accessors
    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        """``cfg.get("retrieval.k_name_char")``"""
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @property
    def data_root(self) -> Path: return self.resolved["data_root"]
    @property
    def output_root(self) -> Path: return self.resolved["output_root"]
    @property
    def runs_root(self) -> Path: return self.resolved["runs_root"]
    @property
    def cache_root(self) -> Path: return self.resolved["cache_root"]
    @property
    def releases_root(self) -> Path: return self.resolved["releases_root"]
    @property
    def project_root(self) -> Path: return self.resolved["project_root"]
    @property
    def validator(self) -> Path: return self.resolved["validator"]
    @property
    def seed(self) -> int: return int(self.raw.get("seed", 42))

    def hash(self) -> str:
        """Stable short hash of the configuration content (for run manifests / cache keys)."""
        return hashlib.sha256(json.dumps(self.raw, sort_keys=True, default=str).encode()).hexdigest()[:12]

    def with_overrides(self, **kv: Any) -> "Config":
        raw = copy.deepcopy(self.raw)
        for k, v in kv.items():
            node = raw
            parts = k.split(".")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = v
        return validate(raw, self.base_dir, self.path)

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.raw, f, sort_keys=False)


PATH_KEYS = ("project_root", "data_root", "output_root", "runs_root", "cache_root", "releases_root", "validator")


def load(path: str | Path) -> Config:
    path = Path(path).resolve()
    if not path.exists():
        raise ConfigError(f"config not found: {path}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    base_dir = path.parent.parent if path.parent.name == "configs" else path.parent
    return validate(raw, base_dir, path)


def validate(raw: dict[str, Any], base_dir: Path, path: Path | None = None) -> Config:
    """Validate types/ranges and resolve paths. Raises ConfigError with an actionable message."""
    if not isinstance(raw, dict):
        raise ConfigError("config must be a mapping")
    cfg = Config(raw=raw, base_dir=Path(base_dir).resolve(), path=path)
    for k in PATH_KEYS:
        v = raw.get(k)
        if not isinstance(v, str) or not v:
            raise ConfigError(f"config key '{k}' must be a non-empty path string")
        p = Path(v)
        cfg.resolved[k] = (cfg.base_dir / p).resolve() if not p.is_absolute() else p
    # never allow outputs to land inside the dataset
    for k in ("output_root", "runs_root", "cache_root", "releases_root"):
        try:
            cfg.resolved[k].relative_to(cfg.data_root)
            raise ConfigError(f"'{k}' must not be inside data_root (would risk overwriting the dataset)")
        except ValueError:
            pass
    if not isinstance(raw.get("seed", 42), int):
        raise ConfigError("seed must be an int")
    r = raw.get("retrieval", {})
    for k in ("k_name_char", "k_addr_char", "k_combined_word", "k_dense", "exact_block_cap", "union_budget", "query_chunk"):
        if not isinstance(r.get(k), int) or r.get(k) <= 0:
            raise ConfigError(f"retrieval.{k} must be a positive int")
    for k in ("min_name_cos", "min_addr_cos", "min_comb_cos"):
        v = r.get(k)
        if not isinstance(v, (int, float)) or not 0 <= v <= 1:
            raise ConfigError(f"retrieval.{k} must be in [0,1]")
    s = raw.get("split", {})
    if not (0 <= float(s.get("holdout_fraction", 0.15)) < 0.5):
        raise ConfigError("split.holdout_fraction must be in [0, 0.5)")
    if int(s.get("n_folds", 5)) < 2:
        raise ConfigError("split.n_folds must be >= 2")
    d = raw.get("decision", {})
    g = d.get("grid", [0.05, 0.96, 0.01])
    if not (isinstance(g, list) and len(g) == 3 and 0 < g[0] < g[1] <= 1 and 0 < g[2] < 1):
        raise ConfigError("decision.grid must be [start, stop, step] with 0<start<stop<=1")
    if d.get("policy", "threshold") not in ("threshold", "threshold_per_source"):
        raise ConfigError("decision.policy must be threshold | threshold_per_source")
    t = raw.get("training", {})
    if t.get("model", "catboost") not in ("catboost", "lightgbm"):
        raise ConfigError("training.model must be catboost | lightgbm")
    c = raw.get("calibration", {})
    if c.get("method", "sigmoid") not in ("sigmoid", "isotonic", "none"):
        raise ConfigError("calibration.method must be sigmoid | isotonic | none")
    for k in ("runs_root", "cache_root", "output_root", "releases_root"):
        cfg.resolved[k].mkdir(parents=True, exist_ok=True)
    return cfg
