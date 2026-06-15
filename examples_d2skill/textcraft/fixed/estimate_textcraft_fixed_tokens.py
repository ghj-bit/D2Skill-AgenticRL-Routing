#!/usr/bin/env python3
"""
Estimate input/output token accounting for TextCraft fixed-route eval results.

The original fixed eval files store API cost but not prompt/completion token usage.
This script reconstructs each model call from saved trajectory conversations and
adds per-trajectory and per-depth estimated input/output token accounting.
"""
import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

RESULT_FILE = "fixed_eval_validation_result_step0.json"

API_PRICE_1M_TOKENS = {
    "qwen3-8B": {"input": 0.050, "output": 0.200},
    "qwen3-30B": {"input": 0.090, "output": 0.300},
    "deepseek": {"input": 0.252, "output": 0.378},
    "deepseek-v3.2": {"input": 0.252, "output": 0.378},
}

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", re.UNICODE)


def estimate_tokens(text: str, method: str = "regex") -> int:
    if not text:
        return 0
    if method == "char4":
        return max(1, math.ceil(len(text) / 4))
    return len(TOKEN_PATTERN.findall(text))


def normalize_model_name(model: str) -> str:
    m = str(model or "").strip()
    lower = m.lower()
    if "deepseek" in lower:
        return "deepseek"
    if "qwen3-30" in lower or "30b" in lower:
        return "qwen3-30B"
    if "qwen3-8" in lower or "8b" in lower or lower == "qwen3-8b":
        return "qwen3-8B"
    return m


def price_for_model(model: str) -> Dict[str, float]:
    return API_PRICE_1M_TOKENS.get(normalize_model_name(model), {"input": 0.0, "output": 0.0})


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def trajectory_path_for_item(run_dir: Path, item_id: str) -> Path:
    return run_dir / "trajectories" / f"{item_id}.json"


def assistant_call_indices(conversations: List[Dict[str, Any]]) -> List[int]:
    assistant_indices = [i for i, msg in enumerate(conversations) if msg.get("role") == "assistant"]
    # The first assistant message is the fixed acknowledgement inserted before the task starts.
    return assistant_indices[1:]


def estimate_trajectory_tokens(conversations: List[Dict[str, Any]], method: str) -> Dict[str, Any]:
    message_texts = [str(msg.get("content", "") or "") for msg in conversations]
    message_tokens = [estimate_tokens(text, method=method) for text in message_texts]
    calls = []
    total_input = 0
    total_output = 0
    for call_number, idx in enumerate(assistant_call_indices(conversations), start=1):
        input_tokens = sum(message_tokens[:idx])
        output_tokens = message_tokens[idx]
        total_input += input_tokens
        total_output += output_tokens
        calls.append(
            {
                "call_index": call_number,
                "conversation_index": idx,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }
        )
    return {
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_tokens": total_input + total_output,
        "num_model_calls": len(calls),
        "calls": calls,
    }


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def summarize_numeric(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "sum": 0.0, "mean": 0.0, "min": None, "max": None}
    return {
        "count": len(values),
        "sum": float(sum(values)),
        "mean": float(sum(values) / len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def update_result_file(result_path: Path, method: str, write_call_details: bool) -> Dict[str, Any]:
    result = read_json(result_path)
    run_dir = result_path.parent
    model = normalize_model_name(str(result.get("model") or run_dir.name))
    prices = price_for_model(model)
    input_price = float(prices.get("input", 0.0))
    output_price = float(prices.get("output", 0.0))

    per_env = result.get("per_env_results", {})
    per_env_records = []
    missing = []
    for env_key, env_result in per_env.items():
        item_id = str(env_result.get("item_id") or "")
        if not item_id:
            missing.append({"env_key": env_key, "reason": "missing item_id"})
            continue
        traj_path = trajectory_path_for_item(run_dir, item_id)
        if not traj_path.exists():
            missing.append({"env_key": env_key, "item_id": item_id, "reason": f"missing trajectory: {traj_path}"})
            continue
        traj = read_json(traj_path)
        token_info = estimate_trajectory_tokens(traj.get("conversations", []), method=method)
        input_cost = token_info["input_tokens"] * input_price / 1_000_000
        output_cost = token_info["output_tokens"] * output_price / 1_000_000
        estimated_cost = input_cost + output_cost
        record = {
            "env_key": env_key,
            "item_id": item_id,
            "depth": str(env_result.get("depth") or traj.get("depth") or "unknown"),
            "input_tokens": token_info["input_tokens"],
            "output_tokens": token_info["output_tokens"],
            "total_tokens": token_info["total_tokens"],
            "num_model_calls": token_info["num_model_calls"],
            "input_token_price_per_1m": input_price,
            "output_token_price_per_1m": output_price,
            "estimated_input_cost": input_cost,
            "estimated_output_cost": output_cost,
            "estimated_api_cost": estimated_cost,
        }
        if write_call_details:
            record["calls"] = token_info["calls"]
        env_result["token_accounting"] = {k: v for k, v in record.items() if k not in {"env_key", "depth"}}
        per_env_records.append(record)

    totals = {
        "input_tokens": sum(r["input_tokens"] for r in per_env_records),
        "output_tokens": sum(r["output_tokens"] for r in per_env_records),
        "total_tokens": sum(r["total_tokens"] for r in per_env_records),
        "num_model_calls": sum(r["num_model_calls"] for r in per_env_records),
        "estimated_input_cost": sum(r["estimated_input_cost"] for r in per_env_records),
        "estimated_output_cost": sum(r["estimated_output_cost"] for r in per_env_records),
        "estimated_api_cost": sum(r["estimated_api_cost"] for r in per_env_records),
    }
    env_count = len(per_env_records) or 1
    depth_groups: Dict[str, List[Dict[str, Any]]] = {}
    for record in per_env_records:
        depth_groups.setdefault(record["depth"], []).append(record)

    depth_summary = {}
    for depth, records in sorted(depth_groups.items(), key=lambda item: (not item[0].isdigit(), int(item[0]) if item[0].isdigit() else item[0])):
        depth_summary[depth] = {
            "total": len(records),
            "input_tokens": sum(r["input_tokens"] for r in records),
            "output_tokens": sum(r["output_tokens"] for r in records),
            "total_tokens": sum(r["total_tokens"] for r in records),
            "num_model_calls": sum(r["num_model_calls"] for r in records),
            "avg_input_tokens": summarize_numeric([r["input_tokens"] for r in records])["mean"],
            "avg_output_tokens": summarize_numeric([r["output_tokens"] for r in records])["mean"],
            "avg_total_tokens": summarize_numeric([r["total_tokens"] for r in records])["mean"],
            "estimated_input_cost": sum(r["estimated_input_cost"] for r in records),
            "estimated_output_cost": sum(r["estimated_output_cost"] for r in records),
            "estimated_api_cost": sum(r["estimated_api_cost"] for r in records),
            "avg_estimated_api_cost": sum(r["estimated_api_cost"] for r in records) / len(records),
        }

    accounting = {
        "method": method,
        "note": "Post-hoc estimate from saved conversation text; original API token usage was not stored.",
        "model": model,
        "input_token_price_per_1m": input_price,
        "output_token_price_per_1m": output_price,
        "env_count": len(per_env_records),
        "missing_trajectories": missing,
        **totals,
        "avg_input_tokens_per_traj": totals["input_tokens"] / env_count,
        "avg_output_tokens_per_traj": totals["output_tokens"] / env_count,
        "avg_total_tokens_per_traj": totals["total_tokens"] / env_count,
        "avg_estimated_api_cost_per_traj": totals["estimated_api_cost"] / env_count,
        "depth_rates": depth_summary,
    }
    result["token_accounting"] = accounting
    result_depth_rates = result.setdefault("depth_rates", {})
    if isinstance(result_depth_rates, dict):
        for depth, token_stats in depth_summary.items():
            depth_entry = result_depth_rates.setdefault(str(depth), {})
            if isinstance(depth_entry, dict):
                depth_entry["token_accounting"] = token_stats

    metrics = result.setdefault("metrics", {})
    metrics["val/token_accounting/input_tokens/total"] = totals["input_tokens"]
    metrics["val/token_accounting/output_tokens/total"] = totals["output_tokens"]
    metrics["val/token_accounting/total_tokens/total"] = totals["total_tokens"]
    metrics["val/token_accounting/input_tokens/mean"] = accounting["avg_input_tokens_per_traj"]
    metrics["val/token_accounting/output_tokens/mean"] = accounting["avg_output_tokens_per_traj"]
    metrics["val/token_accounting/total_tokens/mean"] = accounting["avg_total_tokens_per_traj"]
    metrics["val/token_accounting/estimated_api_cost/total"] = totals["estimated_api_cost"]
    metrics["val/token_accounting/estimated_api_cost/mean"] = accounting["avg_estimated_api_cost_per_traj"]

    write_json(result_path, result)
    return {"path": str(result_path), "env_count": len(per_env_records), **totals}


def find_result_files(root: Path) -> List[Path]:
    if root.is_file() and root.name == RESULT_FILE:
        return [root]
    return sorted(root.glob(f"**/{RESULT_FILE}"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path, help="Result JSON file, model dir, or experiment root")
    parser.add_argument("--method", choices=["regex", "char4"], default="regex")
    parser.add_argument("--write-call-details", action="store_true", help="Store per-call token estimates under each per-env result")
    args = parser.parse_args()

    result_files = []
    for path in args.paths:
        result_files.extend(find_result_files(path))
    seen = set()
    unique_files = []
    for path in result_files:
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique_files.append(path)

    summaries = []
    for result_file in unique_files:
        summaries.append(update_result_file(result_file, method=args.method, write_call_details=args.write_call_details))
        print(f"Updated token accounting: {result_file}")
    print(json.dumps({"updated_files": len(summaries), "summaries": summaries}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
