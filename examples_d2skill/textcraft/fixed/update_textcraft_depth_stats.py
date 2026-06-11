#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_JSON_PATHS = [
    Path(
        "/inspire/hdd/project/ai4education/qianhong-p-qianhong/ghj_workspace/"
        "D2Skill-AgenticRL-Routing/checkpoints/verl_agent_textcraft_fixed_route/"
        "qwen3-8B/fixed_qwen3-8B_seed0/fixed_eval_validation_result_step0.json"
    ),
    Path(
        "/inspire/hdd/project/ai4education/qianhong-p-qianhong/ghj_workspace/"
        "D2Skill-AgenticRL-Routing/checkpoints/verl_agent_textcraft_fixed_route_skills/"
        "qwen3-8B/fixed_qwen3-8B_seed0/fixed_eval_validation_result_step0.json"
    ),
]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def recompute_depth_rates(payload: dict) -> dict:
    per_env_results = payload.get("per_env_results") or {}
    if not isinstance(per_env_results, dict):
        raise ValueError("Expected `per_env_results` to be a dict")

    depth_stats = defaultdict(
        lambda: {
            "success": 0,
            "total": 0,
            "api_cost": 0.0,
            "finished_step_sum": 0.0,
            "finished_step_count": 0,
            "success_items": [],
            "failed_items": [],
        }
    )

    for fallback_id, item in per_env_results.items():
        if not isinstance(item, dict):
            continue
        depth = str(item.get("depth", "unknown"))
        item_id = str(item.get("item_id") or fallback_id)
        won = _as_bool(item.get("won"))
        cost = _as_float(item.get("api_cost"))
        finished_step = item.get("finished_step")

        depth_stats[depth]["total"] += 1
        depth_stats[depth]["success"] += int(won)
        depth_stats[depth]["api_cost"] += cost
        if finished_step is not None:
            depth_stats[depth]["finished_step_sum"] += _as_float(finished_step)
            depth_stats[depth]["finished_step_count"] += 1
        if won:
            depth_stats[depth]["success_items"].append(item_id)
        else:
            depth_stats[depth]["failed_items"].append(item_id)

    depth_rates = {}
    for depth, item in sorted(depth_stats.items(), key=lambda x: x[0]):
        total = int(item["total"])
        if total <= 0:
            continue
        depth_rates[depth] = {
            "success_rate": float(item["success"] / total),
            "success": int(item["success"]),
            "total": total,
            "api_cost": float(item["api_cost"]),
            "avg_api_cost": float(item["api_cost"] / total),
            "avg_finished_step": (
                float(item["finished_step_sum"] / item["finished_step_count"])
                if item["finished_step_count"] > 0
                else None
            ),
            "success_items": list(item["success_items"]),
            "failed_items": list(item["failed_items"]),
        }
    return depth_rates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute TextCraft depth-level api cost and average step stats in fixed eval JSON."
    )
    parser.add_argument("json_path", nargs="*", type=Path, help="JSON files to update. Defaults to fixed-route no-skills and skills files.")
    parser.add_argument("--output", type=Path, default=None, help="Write to another file instead of updating in place. Only valid with one input.")
    args = parser.parse_args()

    json_paths = args.json_path or DEFAULT_JSON_PATHS
    if args.output is not None and len(json_paths) != 1:
        raise ValueError("--output is only valid when updating one JSON file")

    for json_path in json_paths:
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        payload["depth_rates"] = recompute_depth_rates(payload)

        output_path = args.output or json_path
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"Updated depth_rates in {output_path}")


if __name__ == "__main__":
    main()
