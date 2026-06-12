#!/usr/bin/env python3
import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


RESULT_FILE = "fixed_eval_validation_result_step0.json"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _seed_from_dir(path: Path):
    match = re.search(r"_seed(-?\d+)$", path.name)
    return int(match.group(1)) if match else None


def _find_result_files(root: Path) -> list[Path]:
    return sorted(
        root.glob(f"fixed_*_seed*/{RESULT_FILE}"),
        key=lambda p: (_seed_from_dir(p.parent) is None, _seed_from_dir(p.parent) or 0, str(p)),
    )


def _flatten_numbers(value: Any, prefix: str = "") -> dict[str, float]:
    numbers = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            numbers.update(_flatten_numbers(child, child_prefix))
    elif _is_number(value):
        numbers[prefix] = float(value)
    return numbers


def _summarize(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
        "values": values,
    }


def _numeric_summary(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[float]] = {}
    for run in runs:
        for key, value in _flatten_numbers(run["result"]).items():
            grouped.setdefault(key, []).append(value)
    return {key: _summarize(values) for key, values in sorted(grouped.items())}


def _important_summary(numeric_summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    prefixes = [
        "overall_success_rate",
        "env_num",
        "finished_envs",
        "elapsed_seconds",
        "api_cost.total",
        "api_cost.avg_per_traj",
    ]
    summary = {
        key: numeric_summary[key]
        for key in prefixes
        if key in numeric_summary
    }

    depth_summary = {}
    depth_pattern = re.compile(r"^depth_rates\.([^.]+)\.(success_rate|success|total|api_cost|avg_api_cost|avg_finished_step)$")
    for key, stats in numeric_summary.items():
        match = depth_pattern.match(key)
        if not match:
            continue
        depth, metric = match.groups()
        depth_summary.setdefault(depth, {})[metric] = stats
    if depth_summary:
        summary["depth_rates"] = {
            depth: metrics
            for depth, metrics in sorted(
                depth_summary.items(),
                key=lambda item: (not str(item[0]).isdigit(), int(item[0]) if str(item[0]).isdigit() else str(item[0])),
            )
        }

    task_summary = {}
    task_pattern = re.compile(r"^task_rates\.(.+)\.(success_rate|success|total|api_cost|avg_api_cost)$")
    for key, stats in numeric_summary.items():
        match = task_pattern.match(key)
        if not match:
            continue
        task, metric = match.groups()
        task_summary.setdefault(task, {})[metric] = stats
    if task_summary:
        summary["task_rates"] = dict(sorted(task_summary.items()))
    return summary


def aggregate(model_result_root: Path) -> dict[str, Any]:
    result_files = _find_result_files(model_result_root)
    runs = []
    skipped = []
    for result_file in result_files:
        try:
            result = json.loads(result_file.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            skipped.append({"path": str(result_file), "error": str(exc)})
            continue
        runs.append(
            {
                "run_dir": str(result_file.parent),
                "result_file": str(result_file),
                "seed": _seed_from_dir(result_file.parent),
                "result": result,
            }
        )

    numeric = _numeric_summary(runs)
    output = {
        "model_result_root": str(model_result_root),
        "num_runs": len(runs),
        "result_files": [run["result_file"] for run in runs],
        "important_summary": _important_summary(numeric),
        "numeric_summary": numeric,
        "runs": runs,
    }
    if skipped:
        output["skipped"] = skipped
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_result_root", type=Path)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    result = aggregate(args.model_result_root)
    json_out = args.json_out or args.model_result_root / "fixed_eval_validation_result_summary.json"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {json_out}")
    print(f"Aggregated {result['num_runs']} fixed eval result file(s)")


if __name__ == "__main__":
    main()
