#!/usr/bin/env python3
import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Optional


def _finite_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return None


def _summarize(values: list[float]) -> dict:
    stats = {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "stderr": 0.0,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "count": len(values),
        "values": values,
    }
    stats["stderr"] = stats["std"] / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return stats


def _parse_experiment_dir(path: Path) -> tuple[str, Optional[int]]:
    name = path.parent.name
    match = re.fullmatch(r"fixed_(.+)_seed(-?\d+)", name)
    if match:
        return match.group(1), int(match.group(2))
    return name, None


def _collect_files(checkpoint_root: Path, pattern: str) -> list[Path]:
    return sorted(
        path
        for path in checkpoint_root.rglob(pattern)
        if re.fullmatch(r"fixed_.+_seed-?\d+", path.parent.name)
    )


def aggregate(checkpoint_root: Path, pattern: str = "validation_alfworld_task_success_step*.json") -> dict:
    grouped: dict[str, list[dict]] = {}
    skipped = []

    for path in _collect_files(checkpoint_root, pattern):
        model_name, seed = _parse_experiment_dir(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            skipped.append({"path": str(path), "error": str(exc)})
            continue

        run = {
            "path": str(path),
            "experiment_name": payload.get("experiment_name", path.parent.name),
            "global_step": payload.get("global_step"),
            "overall_success_rate": payload.get("overall_success_rate"),
            "alfworld_tasks": payload.get("alfworld_tasks", {}),
        }
        if seed is not None:
            run["seed"] = seed
        grouped.setdefault(model_name, []).append(run)

    result = {}
    for model_name, runs in sorted(grouped.items()):
        runs = sorted(runs, key=lambda item: (item.get("seed", 10**9), str(item.get("path", ""))))

        overall_values = [
            value for value in (_finite_float(run.get("overall_success_rate")) for run in runs)
            if value is not None
        ]

        task_names = sorted(
            {
                task
                for run in runs
                for task in (run.get("alfworld_tasks") or {}).keys()
            }
        )
        task_summary = {}
        for task in task_names:
            metric_names = sorted(
                {
                    metric
                    for run in runs
                    for metric in ((run.get("alfworld_tasks") or {}).get(task) or {}).keys()
                    if _finite_float(((run.get("alfworld_tasks") or {}).get(task) or {}).get(metric)) is not None
                }
            )
            task_metrics = {}
            for metric in metric_names:
                values = [
                    value
                    for value in (
                        _finite_float(((run.get("alfworld_tasks") or {}).get(task) or {}).get(metric))
                        for run in runs
                    )
                    if value is not None
                ]
                if values:
                    task_metrics[metric] = _summarize(values)
            if task_metrics:
                task_summary[task] = task_metrics

        model_result = {
            "num_runs": len(runs),
            "runs": runs,
            "alfworld_tasks": task_summary,
        }
        if overall_values:
            model_result["overall_success_rate"] = _summarize(overall_values)
        result[model_name] = model_result

    if skipped:
        result["_skipped"] = skipped
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate ALFWorld fixed-route task success JSON files under checkpoints."
    )
    parser.add_argument(
        "checkpoint_root",
        nargs="?",
        type=Path,
        default=Path("checkpoints/verl_agent_alfworld_fixed_route"),
        help="Directory containing fixed_<model>_seed* experiment folders.",
    )
    parser.add_argument(
        "--glob",
        default="validation_alfworld_task_success_step*.json",
        help="Metric JSON filename pattern inside each fixed_<model>_seed* folder.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    result = aggregate(args.checkpoint_root, pattern=args.glob)
    if not any(not key.startswith("_") for key in result):
        raise SystemExit(f"No metric JSON files matched under {args.checkpoint_root}")

    json_out = args.json_out or args.checkpoint_root / "validation_alfworld_task_success_summary.json"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {json_out}")
    for model_name, model_result in sorted(result.items()):
        if model_name.startswith("_"):
            continue
        overall = model_result.get("overall_success_rate", {})
        if overall:
            print(
                f"{model_name}: overall_success_rate mean={overall['mean']:.6g} "
                f"std={overall['std']:.6g} count={overall['count']}"
            )
        for task, task_metrics in model_result.get("alfworld_tasks", {}).items():
            success = task_metrics.get("success_rate")
            if success:
                print(
                    f"{model_name}/{task}: success_rate mean={success['mean']:.6g} "
                    f"std={success['std']:.6g} count={success['count']}"
                )


if __name__ == "__main__":
    main()
