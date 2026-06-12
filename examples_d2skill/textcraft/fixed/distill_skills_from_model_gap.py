#!/usr/bin/env python3
"""
Distill TextCraft skills from model-gap trajectories.

This script compares fixed-route trajectories from a weaker model and a stronger
model, selects tasks where the weak model failed but the strong model succeeded,
and asks a DeepSeek/OpenAI-compatible LLM to distill reusable TextCraft skills.

Default use case:
  weak model:   qwen3-8B fixed-route trajectories
  strong model: deepseek-v3.2 fixed-route trajectories

The output is a JSON file with task-level and step-level skills. Each skill is
annotated with source_depth, source_task_id, source_model_pair, and source_files.

The reflection prompt intentionally mirrors agent_system.memory.skill_updater:
  failed trajectory + successful trajectory -> FIRST_ERROR_STEP,
  STEP_REFLECTION, TASK_REFLECTION.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_STRONG_DIR = Path(
    "/inspire/hdd/project/ai4education/qianhong-p-qianhong/ghj_workspace/"
    "D2Skill-AgenticRL-Routing/checkpoints/verl_agent_textcraft_fixed_route/"
    "deepseek-v3.2/fixed_deepseek-v3.2_seed0/trajectories"
)
DEFAULT_WEAK_DIR = Path(
    "/inspire/hdd/project/ai4education/qianhong-p-qianhong/ghj_workspace/"
    "D2Skill-AgenticRL-Routing/checkpoints/verl_agent_textcraft_fixed_route/"
    "qwen3-8B/fixed_qwen3-8B_seed0/trajectories"
)
DEFAULT_OUT = Path("distilled_textcraft_model_gap_skills.json")


@dataclass
class TraceRecord:
    path: Path
    data: Dict[str, Any]
    task: str
    task_key: str
    task_id: str
    depth: Optional[str]
    success: Optional[bool]
    success_reason: str
    turns: List[Dict[str, str]]


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8-sig")
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {"_root": obj}
    except Exception as exc:
        print(f"[skip] failed to read {path}: {exc}", file=sys.stderr)
        return None


def iter_json_files(root: Path) -> Iterable[Path]:
    yield from sorted(p for p in root.rglob("*.json") if p.is_file())


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def stable_hash(text: str, n: int = 16) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:n]


def strip_action(text: str) -> str:
    if not text:
        return ""
    matches = re.findall(r"<action>\s*(.*?)\s*</action>", text, flags=re.I | re.S)
    if matches:
        return normalize_space(matches[-1])
    # Keep raw output short if tags are missing; malformed output is itself useful.
    return normalize_space(text)[:500]


def extract_task_from_text(text: str) -> str:
    if not text:
        return ""
    patterns = [
        r"Task:\s*\n\s*([^\n]+)",
        r"Task:\s*([^\n]+)",
        r"Goal:\s*([^\n]+)",
        r"Your task is:\s*([^\n]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            val = normalize_space(m.group(1))
            if val:
                return val.rstrip(".")
    return ""


def extract_current_observation(prompt: str) -> str:
    if not prompt:
        return ""
    markers = [
        r"Current observation:\s*\n(?P<obs>.*?)(?:\n\nNow it's your turn|\n\nFor the next step|\Z)",
        r"Current observation:\s*(?P<obs>.*?)(?:\n\nNow it's your turn|\n\nFor the next step|\Z)",
    ]
    for pat in markers:
        m = re.search(pat, prompt, flags=re.I | re.S)
        if m:
            return m.group("obs").strip()
    return ""


def extract_task(data: Dict[str, Any], path: Path) -> str:
    for key in ("task", "task_description", "goal", "query"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_space(value).rstrip(".")

    for step in data.get("steps", []) or []:
        if not isinstance(step, dict):
            continue
        for key in ("routed_model_prompt", "router_prompt"):
            task = extract_task_from_text(step.get(key, ""))
            if task:
                return task

    # AgentGym conversation format: initial user message contains Goal.
    for msg in data.get("conversations", []) or []:
        if not isinstance(msg, dict):
            continue
        task = extract_task_from_text(msg.get("content", ""))
        if task:
            return task

    return path.stem


def extract_task_id(data: Dict[str, Any], path: Path, task: str) -> str:
    for key in ("task_id", "uid", "group_uid", "traj_uid", "data_idx", "index"):
        value = data.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    # Prefer trace_533-style numeric ids when available.
    m = re.search(r"(\d+)", path.stem)
    if m:
        return m.group(1)
    return stable_hash(task or path.stem, 12)


def extract_depth(data: Dict[str, Any], path: Path) -> Optional[str]:
    for key in ("depth", "textcraft_depth", "task_depth"):
        value = data.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    for step in data.get("steps", []) or []:
        if not isinstance(step, dict):
            continue
        for key in ("depth", "textcraft_depth", "task_depth"):
            value = step.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        info = step.get("info")
        if isinstance(info, dict):
            value = info.get("depth")
            if value is not None and str(value).strip():
                return str(value).strip()
    m = re.search(r"depth[_-]?(\d+)", str(path), flags=re.I)
    return m.group(1) if m else None


def value_as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value > 0)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "yes", "won", "success", "succeeded", "1"}:
            return True
        if v in {"false", "no", "lost", "fail", "failed", "0"}:
            return False
    return None


def extract_success(data: Dict[str, Any], *, infer_from_text: bool = False) -> Tuple[Optional[bool], str]:
    # Explicit top-level fields first.
    for key in (
        "success",
        "won",
        "is_success",
        "success_rate",
        "success_per_traj",
        "finished_success",
    ):
        if key in data:
            val = value_as_bool(data.get(key))
            if val is not None:
                return val, f"top_level:{key}"

    for key in ("episode_reward", "episode_rewards", "reward", "return", "task_score"):
        if key in data and isinstance(data.get(key), (int, float)):
            return bool(float(data[key]) > 0), f"top_level:{key}>0"

    # Step/info fields, using the final available explicit signal.
    explicit: Optional[Tuple[bool, str]] = None
    for i, step in enumerate(data.get("steps", []) or [], start=1):
        if not isinstance(step, dict):
            continue
        for key in ("success", "won", "done"):
            if key in step:
                val = value_as_bool(step.get(key))
                if val is not None and key != "done":
                    explicit = (val, f"step{i}:{key}")
        info = step.get("info")
        if isinstance(info, dict):
            for key in ("success", "won"):
                if key in info:
                    val = value_as_bool(info.get(key))
                    if val is not None:
                        explicit = (val, f"step{i}.info:{key}")
            for key in ("reward", "task_score"):
                if key in info and isinstance(info.get(key), (int, float)) and info.get(key) > 0:
                    explicit = (True, f"step{i}.info:{key}>0")
        if "reward" in step and isinstance(step.get("reward"), (int, float)) and step.get("reward") > 0:
            explicit = (True, f"step{i}:reward>0")
    if explicit is not None:
        return explicit

    if infer_from_text:
        text = json.dumps(data, ensure_ascii=False).lower()
        success_markers = ["task completed", "success", "you won", "crafted target", "goal achieved"]
        fail_markers = ["failed", "could not", "invalid action", "max steps", "timeout"]
        if any(m in text for m in success_markers) and not any(m in text for m in fail_markers):
            return True, "text_inference"

    return None, "unknown"


def steps_to_turns(data: Dict[str, Any]) -> List[Dict[str, str]]:
    turns: List[Dict[str, str]] = []
    steps = data.get("steps")
    if isinstance(steps, list) and steps:
        for step in steps:
            if not isinstance(step, dict):
                continue
            prompt = step.get("routed_model_prompt") or step.get("router_prompt") or ""
            obs = (
                step.get("observation")
                or step.get("current_observation")
                or step.get("obs")
                or extract_current_observation(prompt)
                or ""
            )
            action = step.get("action") or step.get("routed_model_output") or step.get("model_output") or ""
            action = strip_action(str(action))
            if obs or action:
                turns.append({"observation": str(obs).strip(), "action": action})
        return turns

    # Conversation format: alternate user observations and assistant actions.
    conv = data.get("conversations")
    if isinstance(conv, list) and conv:
        pending_obs = ""
        for msg in conv:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = str(msg.get("content", ""))
            if role == "user":
                pending_obs = content.strip()
            elif role == "assistant":
                action = strip_action(content)
                if pending_obs or action:
                    turns.append({"observation": pending_obs, "action": action})
                    pending_obs = ""
    return turns


def format_turns(turns: List[Dict[str, str]], max_turns: int = 20, max_obs_len: int = 1000) -> str:
    if not turns:
        return "  (no turns found)"
    shown = turns[-max_turns:]
    offset = len(turns) - len(shown)
    lines = []
    if offset > 0:
        lines.append(f"  ... ({offset} earlier turns omitted) ...")
    for i, turn in enumerate(shown, start=offset + 1):
        obs = turn.get("observation", "") or ""
        action = turn.get("action", "") or ""
        if len(obs) > max_obs_len:
            obs = obs[:max_obs_len] + "..."
        lines.append(f"Turn {i}:")
        lines.append(f"  Observation: {obs}")
        lines.append(f"  Action: {action}")
    return "\n".join(lines)


def load_records(root: Path, *, infer_success_from_text: bool = False) -> List[TraceRecord]:
    records: List[TraceRecord] = []
    for path in iter_json_files(root):
        data = load_json(path)
        if data is None:
            continue
        task = extract_task(data, path)
        task_id = extract_task_id(data, path, task)
        task_key = normalize_space(task).lower() or f"id:{task_id}"
        depth = extract_depth(data, path)
        success, reason = extract_success(data, infer_from_text=infer_success_from_text)
        records.append(
            TraceRecord(
                path=path,
                data=data,
                task=task,
                task_key=task_key,
                task_id=task_id,
                depth=depth,
                success=success,
                success_reason=reason,
                turns=steps_to_turns(data),
            )
        )
    return records


def choose_record(records: List[TraceRecord], want_success: bool) -> Optional[TraceRecord]:
    candidates = [r for r in records if r.success is want_success]
    if not candidates:
        return None
    # Prefer shorter successful traces and longer failed traces with more evidence.
    if want_success:
        return sorted(candidates, key=lambda r: (len(r.turns), str(r.path)))[0]
    return sorted(candidates, key=lambda r: (-len(r.turns), str(r.path)))[0]


def find_gap_pairs(weak: List[TraceRecord], strong: List[TraceRecord]) -> Tuple[List[Tuple[TraceRecord, TraceRecord]], Dict[str, Any]]:
    strong_by_task: Dict[str, List[TraceRecord]] = {}
    for rec in strong:
        strong_by_task.setdefault(rec.task_key, []).append(rec)

    pairs: List[Tuple[TraceRecord, TraceRecord]] = []
    stats = {
        "weak_total": len(weak),
        "strong_total": len(strong),
        "weak_failed_known": 0,
        "strong_success_known": 0,
        "weak_unknown_success": 0,
        "strong_unknown_success": 0,
        "unmatched_weak_failures": 0,
    }
    for rec in weak:
        if rec.success is None:
            stats["weak_unknown_success"] += 1
            continue
        if rec.success is not False:
            continue
        stats["weak_failed_known"] += 1
        strong_rec = choose_record(strong_by_task.get(rec.task_key, []), True)
        if strong_rec is None:
            stats["unmatched_weak_failures"] += 1
            continue
        pairs.append((rec, strong_rec))
    for rec in strong:
        if rec.success is None:
            stats["strong_unknown_success"] += 1
        elif rec.success is True:
            stats["strong_success_known"] += 1
    return pairs, stats


def build_reflection_prompt(weak: TraceRecord, strong: TraceRecord, max_turns: int) -> str:
    task_text = weak.task or strong.task or weak.task_id
    depth = weak.depth or strong.depth or "unknown"
    failed_text = format_turns(weak.turns, max_turns=max_turns)
    success_text = format_turns(strong.turns, max_turns=max_turns)
    return f"""You are given one failed TextCraft trajectory from a weaker model and one successful TextCraft trajectory from a stronger model for the same task.
Analyze the difference and produce exactly three outputs.

Task: {task_text}
Task Type: TextCraft crafting
Depth: {depth}
Weak model: qwen3-8B
Strong model: deepseek-v3.2

Failed trajectory from qwen3-8B:
{failed_text}

Successful trajectory from deepseek-v3.2:
{success_text}

Output the following in order (use the exact section headers):

1) FIRST_ERROR_STEP: N
   Where N is the 1-based turn number in the failed trajectory where the weak model first went wrong. Use 0 if unclear.

2) STEP_REFLECTION (one step-level experience for that step only):
   Output a JSON object with exactly: "title", "principle", "when_to_apply".
   This should be a concise TextCraft experience for what to do at that specific step/situation.

Example: {{"title": "Use One Legal Command", "principle": "When gathering resources, issue exactly one valid get command for one item type; do not combine multiple item types in one get command.", "when_to_apply": "When the next step is to collect primitive resources in TextCraft"}}

3) TASK_REFLECTION (one task-level skill for the whole task):
   Output a JSON object with exactly: "title", "principle", "when_to_apply".
   This summarizes for the whole TextCraft task: what to avoid and how to succeed in this kind of crafting task.

Example: {{"title": "Plan Recipe Ledger", "principle": "Before acting, compute the recipe tree, required primitive resources, intermediate outputs, and surplus inventory; execute the plan with legal one-step commands.", "when_to_apply": "When a TextCraft goal requires multi-step crafting or alternative recipe chains"}}

Requirements:
- Skills must be reusable beyond this exact item name.
- Skills must be actionable for a smaller model.
- Do not duplicate the same idea in step and task reflections.
- Do not mention qwen3-8B or deepseek-v3.2 inside the skill text.
- Return JSON objects only under the requested section headers.

Output format (use these exact labels):
FIRST_ERROR_STEP: N

STEP_REFLECTION:
<single JSON object>

TASK_REFLECTION:
<single JSON object>"""


def call_openai_compatible(
    prompt: str,
    *,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    retries: int,
) -> str:
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    url += "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    last_error = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"] or ""
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
            continue
    raise RuntimeError(f"LLM call failed after {retries + 1} attempts: {last_error}")


def extract_json_after_label(raw: str, label: str) -> Optional[Dict[str, Any]]:
    pattern = rf"{re.escape(label)}\s*:\s*(.*?)(?=\n[A-Z_]+\s*:|\Z)"
    m = re.search(pattern, raw, flags=re.I | re.S)
    if not m:
        return None
    text = m.group(1).strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end])
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and obj.get("title") and obj.get("principle"):
        return obj
    return None


def parse_reflection(raw: str) -> Tuple[Optional[int], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    error_turn = None
    m = re.search(r"FIRST_ERROR_STEP\s*:\s*(\d+)", raw, flags=re.I)
    if m:
        error_turn = int(m.group(1))
    step_skill = extract_json_after_label(raw, "STEP_REFLECTION")
    task_skill = extract_json_after_label(raw, "TASK_REFLECTION")
    return error_turn, step_skill, task_skill


def skill_fingerprint(skill: Dict[str, Any]) -> str:
    parts = [
        normalize_space(str(skill.get("title", ""))).lower(),
        normalize_space(str(skill.get("principle", ""))).lower(),
        normalize_space(str(skill.get("when_to_apply", ""))).lower(),
        str(skill.get("source_depth", "")),
    ]
    return "\0".join(parts)


def attach_source_meta(
    skill: Dict[str, Any],
    *,
    kind: str,
    weak: TraceRecord,
    strong: TraceRecord,
    error_turn: Optional[int],
    index: int,
) -> Dict[str, Any]:
    out = dict(skill)
    source_depth = weak.depth or strong.depth or "unknown"
    out.setdefault("when_to_apply", "")
    out["skill_id"] = f"gap_{kind}_{source_depth}_{index:04d}"
    out["source_depth"] = source_depth
    out["source_task_id"] = weak.task_id
    out["source_task"] = weak.task
    out["source_model_pair"] = {"failed": "qwen3-8B", "successful": "deepseek-v3.2"}
    out["source_files"] = {"failed": str(weak.path), "successful": str(strong.path)}
    if error_turn is not None:
        out["source_first_error_step"] = error_turn
    return out


def build_output(
    args: argparse.Namespace,
    stats: Dict[str, Any],
    pair_dump: List[Dict[str, Any]],
    task_skills: List[Dict[str, Any]],
    step_skills: List[Dict[str, Any]],
    raw_records: List[Dict[str, Any]],
    processed_pairs: int,
) -> Dict[str, Any]:
    return {
        "metadata": {
            "weak_dir": str(args.weak_dir),
            "strong_dir": str(args.strong_dir),
            "model": args.model,
            "base_url": args.base_url,
            "selection_stats": stats,
            "selected_pairs": len(pair_dump),
            "processed_pairs": processed_pairs,
            "is_complete": processed_pairs >= len(pair_dump),
            "num_task_skills": len(task_skills),
            "num_step_skills": len(step_skills),
            "num_raw_reflections": len(raw_records),
        },
        "task_skills": task_skills,
        "step_skills": step_skills,
        "pairs": pair_dump,
        "raw_reflections": raw_records,
    }


def write_output_atomic(path: Path, output: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)
def main() -> None:
    parser = argparse.ArgumentParser(description="Distill TextCraft skills from qwen-failed/deepseek-success trajectory pairs.")
    parser.add_argument("--weak-dir", type=Path, default=DEFAULT_WEAK_DIR, help="qwen3-8B trajectory directory")
    parser.add_argument("--strong-dir", type=Path, default=DEFAULT_STRONG_DIR, help="deepseek-v3.2 trajectory directory")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT, help="Output skills JSON path")
    parser.add_argument("--pairs-output", type=Path, default=None, help="Optional JSON file with selected trajectory pairs")
    parser.add_argument("--prompt-dir", type=Path, default=None, help="Optional directory to save LLM prompts and raw responses")
    parser.add_argument("--max-pairs", type=int, default=50, help="Maximum qwen-fail/deepseek-success pairs to distill")
    parser.add_argument("--max-pairs-per-depth", type=int, default=0, help="Optional cap per depth; 0 disables")
    parser.add_argument("--max-turns", type=int, default=18, help="Maximum recent turns included per trajectory")
    parser.add_argument("--dry-run", action="store_true", help="Only select pairs; do not call DeepSeek")
    parser.add_argument("--infer-success-from-text", action="store_true", help="Use text heuristics when explicit success metadata is missing")
    parser.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://api.deepseek.com/v1")
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL") or os.environ.get("OPENAI_MODEL") or "deepseek-chat")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    if not args.weak_dir.exists():
        raise FileNotFoundError(f"weak-dir not found: {args.weak_dir}")
    if not args.strong_dir.exists():
        raise FileNotFoundError(f"strong-dir not found: {args.strong_dir}")

    weak_records = load_records(args.weak_dir, infer_success_from_text=args.infer_success_from_text)
    strong_records = load_records(args.strong_dir, infer_success_from_text=args.infer_success_from_text)
    pairs, stats = find_gap_pairs(weak_records, strong_records)

    if args.max_pairs_per_depth > 0:
        kept = []
        counts: Dict[str, int] = {}
        for weak, strong in pairs:
            depth = weak.depth or strong.depth or "unknown"
            if counts.get(depth, 0) >= args.max_pairs_per_depth:
                continue
            kept.append((weak, strong))
            counts[depth] = counts.get(depth, 0) + 1
        pairs = kept

    pairs = pairs[: max(0, args.max_pairs)]
    pair_dump = [
        {
            "task_id": weak.task_id,
            "task": weak.task,
            "depth": weak.depth or strong.depth,
            "weak_file": str(weak.path),
            "strong_file": str(strong.path),
            "weak_success_reason": weak.success_reason,
            "strong_success_reason": strong.success_reason,
            "weak_turns": len(weak.turns),
            "strong_turns": len(strong.turns),
        }
        for weak, strong in pairs
    ]

    print(json.dumps({"selection_stats": stats, "selected_pairs": len(pairs)}, indent=2, ensure_ascii=False))

    if args.pairs_output:
        args.pairs_output.parent.mkdir(parents=True, exist_ok=True)
        args.pairs_output.write_text(json.dumps(pair_dump, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote pairs: {args.pairs_output}")

    if args.dry_run:
        return
    if not args.api_key:
        raise EnvironmentError("Set DEEPSEEK_API_KEY or pass --api-key before running without --dry-run.")

    if args.prompt_dir:
        args.prompt_dir.mkdir(parents=True, exist_ok=True)

    step_skills: List[Dict[str, Any]] = []
    task_skills: List[Dict[str, Any]] = []
    raw_records: List[Dict[str, Any]] = []
    skill_index = 1
    seen = set()

    for pair_i, (weak, strong) in enumerate(pairs, start=1):
        prompt = build_reflection_prompt(weak, strong, max_turns=args.max_turns)
        if args.prompt_dir:
            (args.prompt_dir / f"pair_{pair_i:04d}_prompt.txt").write_text(prompt, encoding="utf-8")
        try:
            raw = call_openai_compatible(
                prompt,
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                retries=args.retries,
            )
        except Exception as exc:
            print(f"[warn] pair {pair_i} LLM call failed: {exc}", file=sys.stderr)
            raw_records.append({"pair_index": pair_i, "error": str(exc), "pair": pair_dump[pair_i - 1]})
            write_output_atomic(
                args.output,
                build_output(args, stats, pair_dump, task_skills, step_skills, raw_records, processed_pairs=pair_i),
            )
            continue
        if args.prompt_dir:
            (args.prompt_dir / f"pair_{pair_i:04d}_response.txt").write_text(raw, encoding="utf-8")

        error_turn, step_skill, task_skill = parse_reflection(raw)
        raw_records.append({"pair_index": pair_i, "raw_response": raw, "first_error_step": error_turn, "pair": pair_dump[pair_i - 1]})

        for kind, skill, target in (("step", step_skill, step_skills), ("task", task_skill, task_skills)):
            if not skill:
                continue
            with_meta = attach_source_meta(skill, kind=kind, weak=weak, strong=strong, error_turn=error_turn, index=skill_index)
            fp = skill_fingerprint(with_meta)
            if fp in seen:
                continue
            seen.add(fp)
            target.append(with_meta)
            skill_index += 1
        write_output_atomic(
            args.output,
            build_output(args, stats, pair_dump, task_skills, step_skills, raw_records, processed_pairs=pair_i),
        )
        print(f"[{pair_i}/{len(pairs)}] skills: task={len(task_skills)} step={len(step_skills)} wrote={args.output}", flush=True)

    write_output_atomic(
        args.output,
        build_output(args, stats, pair_dump, task_skills, step_skills, raw_records, processed_pairs=len(pairs)),
    )
    print(f"Wrote skills: {args.output}")


if __name__ == "__main__":
    main()


