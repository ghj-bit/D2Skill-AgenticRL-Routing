#!/usr/bin/env python3
"""Run direct fixed-model evaluation on ALFWorld."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple

from omegaconf import OmegaConf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILLRL_ENV_PATH = PROJECT_ROOT / "SkillRL" / "agent_system" / "environments"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import agent_system.environments as environments_pkg  # noqa: E402

if SKILLRL_ENV_PATH.exists() and str(SKILLRL_ENV_PATH) not in environments_pkg.__path__:
    environments_pkg.__path__.append(str(SKILLRL_ENV_PATH))

from agent_system.environments.env_manager import AlfWorldEnvironmentManager  # noqa: E402
from agent_system.environments.env_package.alfworld import (  # noqa: E402
    alfworld_projection,
    build_alfworld_envs,
)
from routing.models_config.models_config import MODEL_CONF  # noqa: E402


ALFWORLD_TASKS = [
    "pick_and_place",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
]


def resolve_model(model: str, api_base: str | None, api_key: str | None) -> Tuple[str, str, str]:
    normalized = str(model or "").strip().lower()
    if "qwen3" in normalized or normalized in {"qwen", "qwen-8b", "qwen8b"}:
        resolved_model = "qwen3-8B"
    elif "deepseek" in normalized:
        resolved_model = "deepseek"
    elif "gemma-2-27b" in normalized or normalized in {"gemma", "gemma-27b", "gemma27b"}:
        resolved_model = "gemma-2-27B"
    else:
        resolved_model = model

    model_conf = MODEL_CONF.get(resolved_model, {})
    base_url = api_base or model_conf.get("api_base") or os.environ.get("OPENAI_BASE_URL", "")
    key = api_key or model_conf.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
    if not base_url:
        raise ValueError(f"No api_base found for model {model!r}. Pass --api-base or add MODEL_CONF.")
    if not key:
        raise ValueError(f"No api_key found for model {model!r}. Pass --api-key or add MODEL_CONF.")
    return resolved_model, base_url, key


class FixedModelAgent:
    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        timeout: int,
        max_retries: int,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        try:
            import openai
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The fixed-model eval script requires the `openai` package for "
                "OpenAI-compatible API calls. Install it in the runtime environment."
            ) from exc
        self.client = openai.OpenAI(
            base_url=api_base,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    def act(self, prompt: str) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )
            content = completion.choices[0].message.content
            return content or "<think>No response.</think><action>pass</action>"
        except Exception as exc:
            print(f"[fixed_eval] API error: {exc}", flush=True)
            return "<think>API request failed.</think><action>pass</action>"


def build_config(args: argparse.Namespace) -> Any:
    return OmegaConf.create(
        {
            "data": {
                "train_batch_size": args.env_num,
                "val_batch_size": args.env_num,
            },
            "env": {
                "env_name": "alfworld/AlfredTWEnv",
                "seed": args.seed,
                "max_steps": args.max_steps,
                "history_length": args.history_length,
                "rollout": {"n": 1},
                "resources_per_worker": {
                    "num_cpus": args.num_cpus_per_worker,
                    "num_gpus": 0.0,
                },
                "alfworld": {"eval_dataset": args.eval_dataset},
                "use_skills_only_memory": False,
                "use_retrieval_memory": False,
            },
        }
    )


def build_env_manager(args: argparse.Namespace, config: Any) -> AlfWorldEnvironmentManager:
    alf_config_path = SKILLRL_ENV_PATH / "env_package" / "alfworld" / "configs" / "config_tw.yaml"
    if not alf_config_path.exists():
        alf_config_path = PROJECT_ROOT / "agent_system" / "environments" / "env_package" / "alfworld" / "configs" / "config_tw.yaml"
    if not alf_config_path.exists():
        raise FileNotFoundError(f"Cannot find ALFWorld config_tw.yaml under {PROJECT_ROOT}")

    envs = build_alfworld_envs(
        str(alf_config_path),
        args.seed + 1000,
        args.env_num,
        1,
        resources_per_worker=OmegaConf.to_container(config.env.resources_per_worker, resolve=True),
        is_train=False,
        env_kwargs={"eval_dataset": args.eval_dataset},
    )
    return AlfWorldEnvironmentManager(envs, partial(alfworld_projection), config)


def task_name_from_gamefile(gamefile: str | None) -> str | None:
    if not gamefile:
        return None
    for task in ALFWORLD_TASKS:
        if task in gamefile:
            return task
    return None


def mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return float(math.sqrt(sum((x - m) ** 2 for x in values) / len(values)))


def summarize_round(wins: List[float], task_wins: Dict[str, List[float]]) -> Dict[str, Any]:
    task_summary = {}
    for task in ALFWORLD_TASKS:
        vals = task_wins.get(task, [])
        task_summary[task] = {
            "success_rate": mean(vals),
            "count": len(vals),
            "success_count": int(sum(vals)),
        }
    return {
        "overall_success_rate": mean(wins),
        "count": len(wins),
        "success_count": int(sum(wins)),
        "tasks": task_summary,
    }


def run_one_round(
    round_idx: int,
    args: argparse.Namespace,
    agent: FixedModelAgent,
    model_name: str,
) -> Dict[str, Any]:
    config = build_config(args)
    env_manager = build_env_manager(args, config)
    observations, _, infos = env_manager.reset({})
    done = [False] * args.env_num
    wins = [0.0] * args.env_num
    final_infos: List[Dict[str, Any]] = [{} for _ in range(args.env_num)]

    try:
        for step in tqdm(range(args.max_steps), desc=f"round {round_idx}", leave=False):
            prompts = observations["text"]
            actions = ["<think>Episode already ended.</think><action>pass</action>"] * args.env_num
            pending = [i for i, is_done in enumerate(done) if not is_done]

            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                generated = list(executor.map(agent.act, [prompts[i] for i in pending]))
            for env_idx, action in zip(pending, generated):
                actions[env_idx] = action

            observations, _, rewards, dones, infos = env_manager.step(
                actions,
                models=[model_name] * args.env_num,
            )
            for i, is_done in enumerate(dones):
                if is_done and not done[i]:
                    done[i] = True
                    wins[i] = float(infos[i].get("won", rewards[i] > 0))
                    final_infos[i] = dict(infos[i])

            print(
                f"[fixed_eval] round={round_idx} step={step + 1} "
                f"done={sum(done)}/{args.env_num} success={mean([wins[i] for i, d in enumerate(done) if d]):.4f}",
                flush=True,
            )
            if all(done):
                break

        for i, is_done in enumerate(done):
            if not is_done:
                final_infos[i] = dict(infos[i])
                wins[i] = float(infos[i].get("won", 0.0))
    finally:
        close = getattr(env_manager.envs, "close", None)
        if callable(close):
            close()

    task_wins = {task: [] for task in ALFWORLD_TASKS}
    unknown = []
    for win, info in zip(wins, final_infos):
        task = task_name_from_gamefile(info.get("extra.gamefile"))
        if task is None:
            unknown.append(info.get("extra.gamefile"))
            continue
        task_wins[task].append(win)

    summary = summarize_round(wins, task_wins)
    summary["round"] = round_idx
    summary["unknown_gamefiles"] = unknown
    return summary


def aggregate(rounds: List[Dict[str, Any]]) -> Dict[str, Any]:
    overall = [r["overall_success_rate"] for r in rounds]
    tasks = {}
    for task in ALFWORLD_TASKS:
        values = [r["tasks"][task]["success_rate"] for r in rounds if r["tasks"][task]["count"] > 0]
        counts = [r["tasks"][task]["count"] for r in rounds]
        successes = [r["tasks"][task]["success_count"] for r in rounds]
        tasks[task] = {
            "mean_success_rate": mean(values),
            "std_success_rate": std(values),
            "round_count": len(values),
            "total_count": int(sum(counts)),
            "total_success_count": int(sum(successes)),
        }
    return {
        "overall_mean_success_rate": mean(overall),
        "overall_std_success_rate": std(overall),
        "round_count": len(rounds),
        "tasks": tasks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("FIXED_EVAL_MODEL", "deepseek-v3.2"))
    parser.add_argument("--api-base", default=os.environ.get("FIXED_EVAL_API_BASE"))
    parser.add_argument("--api-key", default=os.environ.get("FIXED_EVAL_API_KEY"))
    parser.add_argument("--env-num", type=int, default=int(os.environ.get("FIXED_EVAL_ENV_NUM", 200)))
    parser.add_argument("--test-times", type=int, default=int(os.environ.get("FIXED_EVAL_TEST_TIMES", 3)))
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("FIXED_EVAL_MAX_STEPS", 50)))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("FIXED_EVAL_SEED", 0)))
    parser.add_argument("--eval-dataset", default=os.environ.get("FIXED_EVAL_DATASET", "eval_in_distribution"))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("FIXED_EVAL_TEMPERATURE", 0.4)))
    parser.add_argument("--top-p", type=float, default=float(os.environ.get("FIXED_EVAL_TOP_P", 1.0)))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("FIXED_EVAL_MAX_TOKENS", 256)))
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("FIXED_EVAL_TIMEOUT", 120)))
    parser.add_argument("--max-retries", type=int, default=int(os.environ.get("FIXED_EVAL_MAX_RETRIES", 2)))
    parser.add_argument("--max-workers", type=int, default=int(os.environ.get("FIXED_EVAL_MAX_WORKERS", 8)))
    parser.add_argument("--history-length", type=int, default=int(os.environ.get("FIXED_EVAL_HISTORY_LENGTH", 4)))
    parser.add_argument("--num-cpus-per-worker", type=float, default=float(os.environ.get("FIXED_EVAL_CPUS_PER_WORKER", 0.05)))
    parser.add_argument("--output-dir", default=os.environ.get("FIXED_EVAL_OUTPUT_DIR"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_name, api_base, api_key = resolve_model(args.model, args.api_base, args.api_key)
    agent = FixedModelAgent(
        model=model_name,
        api_base=api_base,
        api_key=api_key,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or PROJECT_ROOT / "outputs" / "fixed_model_alfworld_eval" / f"{model_name}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rounds = []
    start_time = time.time()
    for round_idx in range(args.test_times):
        summary = run_one_round(round_idx, args, agent, model_name)
        rounds.append(summary)
        round_path = output_dir / f"round_{round_idx}_summary.json"
        round_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[fixed_eval] wrote {round_path}", flush=True)

    result = {
        "model": model_name,
        "args": vars(args),
        "elapsed_seconds": time.time() - start_time,
        "aggregate": aggregate(rounds),
        "rounds": rounds,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2, ensure_ascii=False), flush=True)
    print(f"[fixed_eval] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
