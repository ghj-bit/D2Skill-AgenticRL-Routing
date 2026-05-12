#!/usr/bin/env python3
import argparse
import ast
import csv
import json
import math
import statistics
from pathlib import Path


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
                return ast.literal_eval(raw)

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
    for log_path in sorted(log_root.glob("*/seed_*.log")):
        model_name = log_path.parent.name
        metrics = _numeric_items(_extract_metric_dict(log_path.read_text(encoding="utf-8", errors="ignore")))
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
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_root", type=Path)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--csv-out", type=Path, default=None)
    args = parser.parse_args()

    result = aggregate(args.log_root)
    json_out = args.json_out or args.log_root / "fixed_route_metric_summary.json"
    csv_out = args.csv_out or args.log_root / "fixed_route_metric_summary.csv"

    json_out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "metric", "mean", "std", "count", "values"])
        for model_name, model_result in sorted(result.items()):
            for metric, stats in sorted(model_result["metrics"].items()):
                writer.writerow([
                    model_name,
                    metric,
                    stats["mean"],
                    stats["std"],
                    stats["count"],
                    json.dumps(stats["values"]),
                ])

    print(f"Wrote {json_out}")
    print(f"Wrote {csv_out}")


if __name__ == "__main__":
    main()
