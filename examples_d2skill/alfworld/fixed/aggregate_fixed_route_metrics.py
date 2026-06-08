#!/usr/bin/env python3
import argparse
import ast
import json
import math
import os
import re
import statistics
from pathlib import Path
from typing import Optional


ALFWORLD_TASKS = [
    "pick_and_place",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
]


def _extract_numeric_pairs(text: str) -> dict:
    """Extract numeric metric entries from a printed dict, ignoring text fields."""
    metrics = {}
    pattern = re.compile(
        r"""(?P<quote>['"])(?P<key>(?:\\.|(?!\1).)*?)(?P=quote)\s*:\s*"""
        r"""(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"""
    )
    for match in pattern.finditer(text):
        value = float(match.group("value"))
        if math.isfinite(value):
            metrics[match.group("key")] = value
    return metrics


def _extract_metric_dict(text: str) -> dict:
    marker_candidates = ["Initial validation metrics:", "Final validation metrics:"]
    marker_pos = -1
    marker = ""
    for candidate in marker_candidates:
        pos = text.rfind(candidate)
        if pos > marker_pos:
            marker_pos = pos
            marker = candidate
    if marker_pos < 0:
        raise ValueError("No validation metrics marker found")

    start = text.find("{", marker_pos + len(marker))
    if start < 0:
        raise ValueError("No metrics dict found after validation metrics marker")

    parse_error = None
    depth = 0
    in_string = False
    string_quote = ""
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == string_quote:
                in_string = False
            continue
        if ch in ("'", '"'):
            in_string = True
            string_quote = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                raw = text[start:idx + 1]
                try:
                    return ast.literal_eval(raw)
                except (SyntaxError, ValueError) as exc:
                    parse_error = exc
                    break

    fallback = _extract_numeric_pairs(text[start:])
    if fallback:
        return fallback

    if parse_error is not None:
        raise ValueError(f"Failed to parse metrics dict: {parse_error}") from parse_error
    raise ValueError("Unclosed metrics dict")


def _numeric_items(metrics: dict) -> dict:
    out = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            value = float(value)
            if math.isfinite(value):
                out[key] = value
    return out


def _summarize_values(values: list[float]) -> dict:
    stats = {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "values": values,
    }
    stats["median"] = statistics.median(values)
    stats["stderr"] = stats["std"] / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return stats


def _metric_summary(runs: list[dict], metric_name: str) -> Optional[dict]:
    values = [run["metrics"][metric_name] for run in runs if metric_name in run["metrics"]]
    if not values:
        return None
    return _summarize_values(values)


def _build_alfworld_task_summary(runs: list[dict]) -> dict:
    task_summary = {}
    for task in ALFWORLD_TASKS:
        task_metrics = {}

        success_metric = f"val/{task}_success_rate"
        success_stats = _metric_summary(runs, success_metric)
        success_source = success_metric

        # Older logs may only contain val/<task>/test_score. In ALFWorld this is
        # the mean binary episode reward, so it is a usable success-rate fallback.
        if success_stats is None:
            fallback_metric = f"val/{task}/test_score"
            success_stats = _metric_summary(runs, fallback_metric)
            success_source = fallback_metric

        if success_stats is not None:
            task_metrics["success_rate"] = {
                "source_metric": success_source,
                **success_stats,
            }

        for output_name, metric_name in (
            ("test_score", f"val/{task}/test_score"),
            ("tool_call_count_mean", f"val/{task}/tool_call_count/mean"),
        ):
            stats = _metric_summary(runs, metric_name)
            if stats is not None:
                task_metrics[output_name] = {
                    "source_metric": metric_name,
                    **stats,
                }

        if task_metrics:
            task_summary[task] = task_metrics

    return task_summary


def _build_run_metrics(log_path: str, metrics: dict) -> dict:
    run_record = {
        "path": log_path,
        "metrics": {},
        "alfworld_tasks": {},
    }
    seed_match = re.search(r"seed_(-?\d+)\.log$", Path(log_path).name)
    if seed_match:
        run_record["seed"] = int(seed_match.group(1))

    if "val/success_rate" in metrics:
        run_record["metrics"]["success_rate"] = {
            "source_metric": "val/success_rate",
            "value": metrics["val/success_rate"],
        }

    for task in ALFWORLD_TASKS:
        task_metrics = {}
        success_metric = f"val/{task}_success_rate"
        fallback_metric = f"val/{task}/test_score"
        if success_metric in metrics:
            task_metrics["success_rate"] = {
                "source_metric": success_metric,
                "value": metrics[success_metric],
            }
        elif fallback_metric in metrics:
            task_metrics["success_rate"] = {
                "source_metric": fallback_metric,
                "value": metrics[fallback_metric],
            }

        for output_name, metric_name in (
            ("test_score", fallback_metric),
            ("tool_call_count_mean", f"val/{task}/tool_call_count/mean"),
        ):
            if metric_name in metrics:
                task_metrics[output_name] = {
                    "source_metric": metric_name,
                    "value": metrics[metric_name],
                }

        if task_metrics:
            run_record["alfworld_tasks"][task] = task_metrics

    if not run_record["metrics"]:
        run_record.pop("metrics")
    if not run_record["alfworld_tasks"]:
        run_record.pop("alfworld_tasks")
    return run_record


def aggregate(log_root: Path) -> dict:
    grouped = {}
    skipped = []
    for log_path in sorted(log_root.glob("*/seed_*.log")):
        model_name = log_path.parent.name
        try:
            metrics = _numeric_items(_extract_metric_dict(log_path.read_text(encoding="utf-8", errors="ignore")))
        except Exception as exc:
            skipped.append({"path": str(log_path), "error": str(exc)})
            print(f"Warning: skipped {log_path}: {exc}")
            continue
        grouped.setdefault(model_name, []).append({"path": str(log_path), "metrics": metrics})

    result = {}
    for model_name, runs in grouped.items():
        keys = sorted(set().union(*(run["metrics"].keys() for run in runs)))
        summary = {}
        for key in keys:
            values = [run["metrics"][key] for run in runs if key in run["metrics"]]
            if not values:
                continue
            summary[key] = _summarize_values(values)
        model_result = {
            "num_runs": len(runs),
            "runs": [run["path"] for run in runs],
            "run_metrics": [
                _build_run_metrics(run["path"], run["metrics"])
                for run in runs
            ],
            "metrics": summary,
        }
        alfworld_tasks = _build_alfworld_task_summary(runs)
        if alfworld_tasks:
            model_result["alfworld_tasks"] = alfworld_tasks
        result[model_name] = model_result
    if skipped:
        result["_skipped"] = skipped
    return result


def _get_stats(result: dict, model_name: str, metric_name: str) -> Optional[dict]:
    return result.get(model_name, {}).get("metrics", {}).get(metric_name)


def _find_success_rate_metric(model_result: dict) -> Optional[str]:
    metrics = model_result.get("metrics", {})
    if "val/success_rate" in metrics:
        return "val/success_rate"

    for metric in sorted(metrics):
        if metric.endswith("/success_rate"):
            return metric

    for metric in sorted(metrics):
        if "success_rate" in metric:
            return metric

    return None


def build_wandb_summary(result: dict) -> dict:
    summary = {}
    for model_name, model_result in sorted(result.items()):
        if model_name.startswith("_") or not isinstance(model_result, dict):
            continue
        model_result = result.get(model_name)
        if not model_result:
            continue

        metric_name = _find_success_rate_metric(model_result)
        if metric_name is None:
            continue

        stats = _get_stats(result, model_name, metric_name)
        if stats is None:
            continue

        summary[f"{model_name}/val_success_rate_mean"] = stats["mean"]
        summary[f"{model_name}/val_success_rate_std"] = stats["std"]

        for task, task_metrics in model_result.get("alfworld_tasks", {}).items():
            task_success = task_metrics.get("success_rate")
            if not task_success:
                continue
            summary[f"{model_name}/{task}/val_success_rate_mean"] = task_success["mean"]
            summary[f"{model_name}/{task}/val_success_rate_std"] = task_success["std"]
    return summary


def log_wandb_summary(summary: dict, project: str, name: str) -> None:
    if not summary:
        print("No success_rate summary found; skipped wandb logging")
        return

    try:
        import wandb
    except ImportError:
        print("wandb is not installed; skipped wandb logging")
        return

    os.environ.setdefault("WANDB_MODE", "offline")
    run = wandb.init(project=project, name=name)
    wandb.log(summary)
    run.finish()
    print(f"Wrote wandb summary run: project={project} name={name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_root", type=Path)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--wandb-project", default="verl_agent_alfworld_fixed_route")
    parser.add_argument("--wandb-name", default="fixed_route_5x_summary")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    result = aggregate(args.log_root)
    json_out = args.json_out or args.log_root / "fixed_route_metric_summary.json"
    json_out.parent.mkdir(parents=True, exist_ok=True)

    json_out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {json_out}")

    wandb_summary = build_wandb_summary(result)
    if wandb_summary:
        print("Fixed-route success_rate summary:")
        for key, value in sorted(wandb_summary.items()):
            print(f"  {key}: {value}")
    if not args.no_wandb:
        log_wandb_summary(wandb_summary, args.wandb_project, args.wandb_name)


if __name__ == "__main__":
    main()
