#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


DEFAULT_INPUT_DIRS = (
    (
        "/inspire/hdd/project/ai4education/qianhong-p-qianhong/ghj_workspace/"
        "AgentGym-RL/AgentGym/agentenv/examples/basic/outputs/qwen3-8B_textcraft"
    ),
    (
        "/inspire/hdd/project/ai4education/qianhong-p-qianhong/ghj_workspace/"
        "AgentGym-RL/AgentGym/agentenv/examples/basic/outputs/deepseek_textcraft"
    ),
)


def is_success_zero(path: Path) -> bool:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("success") == 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="List JSON files whose top-level success field is 0."
    )
    parser.add_argument(
        "input_dirs",
        nargs="*",
        default=None,
        help=f"Directories containing JSON files. Default: {', '.join(DEFAULT_INPUT_DIRS)}",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search JSON files recursively.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional txt file to save matched JSON file paths.",
    )
    args = parser.parse_args()

    input_dirs = [Path(path) for path in (args.input_dirs or DEFAULT_INPUT_DIRS)]
    for input_dir in input_dirs:
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    pattern = "**/*.json" if args.recursive else "*.json"
    matched = []
    failed = []

    for input_dir in input_dirs:
        for path in sorted(input_dir.glob(pattern)):
            try:
                if is_success_zero(path):
                    matched.append(path)
            except Exception as exc:
                failed.append((path, exc))

    for path in matched:
        print(path)

    print(f"\nMatched success=0 JSON files: {len(matched)}")
    if failed:
        print(f"Skipped invalid/unreadable JSON files: {len(failed)}")
        for path, exc in failed:
            print(f"[SKIP] {path}: {exc}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "\n".join(str(path) for path in matched) + ("\n" if matched else ""),
            encoding="utf-8",
        )
        print(f"Saved matched paths to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
