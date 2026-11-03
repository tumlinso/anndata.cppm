#!/usr/bin/env python3
"""Recreate comparison-4096 fixtures using the benchmark's exact generator."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


if len(sys.argv) != 3:
    raise SystemExit("usage: recreate_fixture.py STORAGE_BENCHMARK_PY CAMPAIGN_ROOT")
benchmark_path = Path(sys.argv[1]).resolve()
campaign_root = Path(sys.argv[2]).resolve()
destination = campaign_root / "comparison-4096"
if destination.exists():
    raise SystemExit(f"refusing to reuse fixture directory: {destination}")

spec = importlib.util.spec_from_file_location("storage_benchmark", benchmark_path)
if spec is None or spec.loader is None:
    raise SystemExit(f"cannot import benchmark generator: {benchmark_path}")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
records = benchmark.generate_fixtures(
    destination,
    size=4096,
    include_huge=False,
    huge_size=524288,
    chunk=1024,
)
print(json.dumps(records, indent=2, sort_keys=True))
