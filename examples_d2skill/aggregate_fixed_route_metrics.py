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
            summary[key] = {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "count": len(values),
                "values": values,
            }
        result[model_name] = {
            "num_runs": len(runs),
            "runs": [run["path"] for run in runs],
            "metrics": summary,
        }
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
    for model_name in ("deepseek", "qwen3-8B"):
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
