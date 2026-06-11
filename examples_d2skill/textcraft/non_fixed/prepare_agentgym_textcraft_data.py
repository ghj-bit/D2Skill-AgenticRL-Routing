#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import pandas as pd


def _read_json_records(path: Path):
    with path.open("r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "train", "eval", "test"):
            if isinstance(payload.get(key), list):
                return payload[key]
        return [payload]
    raise TypeError(f"Unsupported JSON payload in {path}: {type(payload)}")


def _extract_item_id(record, fallback: int):
    if isinstance(record, dict):
        for key in ("item_id", "data_idx", "id", "index"):
            if key in record:
                raw_item_id = record[key]
                break
        else:
            raw_item_id = f"textcraft_{fallback}"
    else:
        raw_item_id = record

    if isinstance(raw_item_id, int):
        return raw_item_id, raw_item_id

    raw_item_id = str(raw_item_id)
    match = re.search(r"(-?\d+)$", raw_item_id)
    data_idx = int(match.group(1)) if match else fallback
    return raw_item_id, data_idx


def _extract_depth(record):
    if not isinstance(record, dict):
        return None
    for key in ("depth", "task_depth", "recipe_depth", "craft_depth"):
        if key in record:
            return record[key]
    for key, value in record.items():
        if "depth" in str(key).lower():
            return value
    return None


def _convert(src: Path, dst: Path, split: str) -> None:
    rows = []
    for idx, record in enumerate(_read_json_records(src)):
        item_id, data_idx = _extract_item_id(record, idx)
        depth = _extract_depth(record)
        env_kwargs = {"data_idx": data_idx}
        extra_info = {
            "split": split,
            "index": idx,
            "item_id": item_id,
            "data_idx": data_idx,
        }
        if depth is not None:
            env_kwargs["depth"] = depth
            extra_info["depth"] = depth
        rows.append(
            {
                "data_source": "textcraft",
                "item_id": item_id,
                "prompt": [{"role": "user", "content": ""}],
                "ability": "agent",
                "env_kwargs": env_kwargs,
                "extra_info": extra_info,
            }
        )
    if not rows:
        raise ValueError(f"No records found in {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(dst, index=False)
    print(f"Wrote {len(rows)} TextCraft rows to {dst}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", required=True, type=Path)
    parser.add_argument("--val-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    _convert(args.train_json, args.output_dir / "train.parquet", "train")
    _convert(args.val_json, args.output_dir / "test.parquet", "test")


if __name__ == "__main__":
    main()
