#!/usr/bin/env python3
import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def iter_json_files(inputs: Iterable[Path], pattern: str) -> List[Path]:
    files: List[Path] = []
    for item in inputs:
        if item.is_dir():
            files.extend(sorted(item.rglob(pattern)))
        elif item.is_file():
            files.append(item)
    return files


def flatten_numeric(obj: Any, prefix: str = "") -> Dict[str, float]:
    values: Dict[str, float] = {}
    if isinstance(obj, bool):
        return values
    if isinstance(obj, (int, float)):
        value = float(obj)
        if math.isfinite(value) and prefix:
            values[prefix] = value
        return values
    if isinstance(obj, dict):
        # Common metric-summary shape:
        # {"metric/name": {"mean": 1.0, "std": 0.1, ...}}
        for key, value in obj.items():
            key = str(key)
            child_prefix = f"{prefix}.{key}" if prefix else key
            values.update(flatten_numeric(value, child_prefix))
        return values
    if isinstance(obj, list):
        # Trajectory dumps are often arrays of records. Aggregate per-file list
        # values so each JSON contributes one value per metric.
        per_key: Dict[str, List[float]] = {}
        for item in obj:
            for key, value in flatten_numeric(item).items():
                per_key.setdefault(key, []).append(value)
        for key, vals in per_key.items():
            if vals:
                values[f"{prefix}.{key}.mean" if prefix else f"{key}.mean"] = statistics.fmean(vals)
        if obj:
            values[f"{prefix}.num_items" if prefix else "num_items"] = float(len(obj))
        return values
    return values


def summarize(files: List[Path]) -> Dict[str, Any]:
    grouped: Dict[str, List[float]] = {}
    runs = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        metrics = flatten_numeric(data)
        runs.append({"path": str(path), "num_metrics": len(metrics)})
        for key, value in metrics.items():
            grouped.setdefault(key, []).append(value)

    summary = {}
    for key in sorted(grouped):
        values = grouped[key]
        summary[key] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "count": len(values),
            "values": values,
        }
    return {"num_files": len(files), "runs": runs, "metrics": summary}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate numeric metrics from fixed-route JSON outputs."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="JSON files or directories")
    parser.add_argument("--glob", default="*.json", help="Pattern used when an input is a directory")
    parser.add_argument(
        "--path-contains",
        default=None,
        help="Keep only JSON files whose path contains this substring, e.g. deepseek",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    files = iter_json_files(args.inputs, args.glob)
    if args.path_contains:
        needle = args.path_contains.lower()
        files = [p for p in files if needle in str(p).lower()]
    files = sorted(dict.fromkeys(files))
    if not files:
        raise SystemExit("No JSON files matched")

    result = summarize(files)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {args.json_out}")

    print(f"Aggregated {result['num_files']} JSON file(s)")
    for key, stats in result["metrics"].items():
        print(
            f"{key}: mean={stats['mean']:.6g} std={stats['std']:.6g} "
            f"count={stats['count']} min={stats['min']:.6g} max={stats['max']:.6g}"
        )


if __name__ == "__main__":
    main()
