#!/usr/bin/env python3
"""Concurrent fixed-model ALFWorld evaluation.

This follows the structure of verl-agent's prompt_agent/gpt4o_alfworld_concurrent.py
while keeping local MODEL_CONF based model resolution.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SKILLRL_ROOT = PROJECT_ROOT / "SkillRL"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if SKILLRL_ROOT.exists() and str(SKILLRL_ROOT) not in sys.path:
    sys.path.append(str(SKILLRL_ROOT))

from routing.models_config.models_config import MODEL_CONF  # noqa: E402


TASKS = [
    "pick_and_place",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
]

API_PRICE_1M_TOKENS = {
    "qwen3-8B": {
        "input": 0.050,
        "output": 0.200,
    },
    "qwen3-30B": {
        "input": 0.090,
        "output": 0.300,
    },
    "deepseek": {
        "input": 0.252,
        "output": 0.378,
    },
}


def empty_token_usage() -> dict:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "input_token_cost": 0.0,
        "output_token_cost": 0.0,
        "token_cost": 0.0,
        "estimated_input_api_cost": 0.0,
        "estimated_output_api_cost": 0.0,
        "estimated_api_cost": 0.0,
    }


def add_token_usage(dst: dict, src: dict) -> None:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        dst[key] = int(dst.get(key, 0)) + int(src.get(key, 0))
    for key in (
        "input_token_cost",
        "output_token_cost",
        "token_cost",
        "estimated_input_api_cost",
        "estimated_output_api_cost",
        "estimated_api_cost",
    ):
        dst[key] = float(dst.get(key, 0.0)) + float(src.get(key, 0.0))


def build_token_usage(usage, model_name: str) -> dict:
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = getattr(usage, "total_tokens", None)
    if total_tokens is None:
        total_tokens = prompt_tokens + completion_tokens
    total_tokens = int(total_tokens or 0)

    prices = API_PRICE_1M_TOKENS.get(model_name, {})
    input_price_per_1m = float(prices.get("input", 0.0))
    output_price_per_1m = float(prices.get("output", 0.0))
    input_token_cost = float(prompt_tokens * input_price_per_1m / 1_000_000)
    output_token_cost = float(completion_tokens * output_price_per_1m / 1_000_000)
    token_cost = input_token_cost + output_token_cost
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "input_token_cost": input_token_cost,
        "output_token_cost": output_token_cost,
        "token_cost": token_cost,
        "estimated_input_api_cost": input_token_cost,
        "estimated_output_api_cost": output_token_cost,
        "estimated_api_cost": token_cost,
    }


def resolve_model(model: str, api_base: str | None, api_key: str | None) -> Tuple[str, str, str]:
    normalized = str(model or "").strip().lower()
    if "qwen3-30b" in normalized or normalized in {"qwen-30b", "qwen30b"}:
        resolved_model = "qwen3-30B"
    elif "qwen3" in normalized or normalized in {"qwen", "qwen-8b", "qwen8b"}:
        resolved_model = "qwen3-8B"
    elif "deepseek" in normalized:
        resolved_model = "deepseek"
    else:
        resolved_model = model

    model_conf = MODEL_CONF.get(resolved_model, {})
    base_url = (
        api_base
        or model_conf.get("api_base")
        or os.environ.get("OPENAI_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
        or ""
    )
    key = api_key or model_conf.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
    if not base_url:
        raise ValueError(f"No api_base found for model {model!r}.")
    if not key:
        raise ValueError(f"No api_key found for model {model!r}.")
    return resolved_model, base_url, key


def build_env(
    env_name: str,
    env_num: int = 1,
    seed: int = 1,
    max_steps: int = 50,
    history_length: int = 2,
    eval_dataset: str = "eval_in_distribution",
    use_skills_only_memory: bool = True,
    skills_json_path: str | None = None,
    skill_retrieval_service_url: str | None = None,
    retrieval_mode: str = "embedding",
    embedding_model_path: str | None = None,
    skill_text_for_retrieval: str = "when_to_apply",
    similarity_threshold: float | None = 0.7,
    top_k_task: int = 3,
    top_k_step: int = 3,
    skill_gen_mode: str = "task_step",
):
    if env_name != "alfworld":
        raise ValueError(f"Unsupported environment name: {env_name}")

    from omegaconf import OmegaConf
    from agent_system.environments.env_manager import make_envs

    config = OmegaConf.create(
        {
            "data": {
                "train_batch_size": 1,
                "val_batch_size": env_num,
            },
            "env": {
                "env_name": "alfworld/AlfredTWEnv",
                "seed": 0,
                "val_seed": seed,
                "max_steps": max_steps,
                "rollout": {"n": 1},
                "resources_per_worker": {"num_cpus": 0.1, "num_gpus": 0.0},
                "alfworld": {"eval_dataset": eval_dataset},
                "history_length": history_length,
                "use_skills_only_memory": use_skills_only_memory,
                "skills_only_memory": {
                    "skills_json_path": skills_json_path,
                    "retrieval_mode": retrieval_mode,
                    "embedding_model_path": embedding_model_path,
                    "skill_retrieval_service_url": skill_retrieval_service_url,
                    "skill_text_for_retrieval": skill_text_for_retrieval,
                    "similarity_threshold": similarity_threshold,
                    "top_k_task": top_k_task,
                    "top_k_step": top_k_step,
                    "skill_gen_mode": skill_gen_mode,
                    "enable_dynamic_management": False,
                },
            },
        }
    )
    train_envs, val_envs = make_envs(config)
    if hasattr(train_envs, "envs") and hasattr(train_envs.envs, "close"):
        train_envs.envs.close()
    return val_envs


class Agent:
    def __init__(
        self,
        model_name: str,
        api_key: str,
        api_base: str,
        temperature: float = 0.4,
        timeout: float = 120,
        max_retries: int = 3,
    ):
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The fixed-model eval script requires the `openai` package."
            ) from exc

        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout,
        )

    def get_action_from_gpt(self, obs: str) -> tuple[str, dict]:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": obs}],
                    temperature=self.temperature,
                    n=1,
                    stop=None,
                )
                usage = build_token_usage(getattr(response, "usage", None), self.model_name)
                return response.choices[0].message.content.strip(), usage
            except Exception as exc:
                last_error = exc
                sleep_s = min(2 ** attempt, 8)
                logging.warning("API request failed on attempt %s: %s", attempt + 1, exc)
                time.sleep(sleep_s)
        raise RuntimeError(f"API request failed after {self.max_retries} attempts") from last_error


def setup_logger(output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    log_fp = os.path.join(output_dir, "run.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_fp, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return log_fp


def infer_actions_concurrently(agent: Agent, text_obs, env_dones, max_workers: int):
    actions = ["None"] * len(text_obs)
    token_usages = [empty_token_usage() for _ in range(len(text_obs))]
    active_indices = [i for i, done in enumerate(env_dones) if not done]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(agent.get_action_from_gpt, text_obs[i]): i
            for i in active_indices
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                action, token_usage = future.result()
                actions[idx] = action
                token_usages[idx] = token_usage
            except Exception as exc:
                logging.exception("Env %s failed to get model action: %s", idx, exc)
                actions[idx] = "<think>API request failed.</think><action>look</action>"

    return actions, token_usages


def task_from_gamefile(gamefile: str) -> str:
    for task in TASKS:
        if task in gamefile:
            return task
    return "other"


def run_one_test(test_idx: int, env_manager, agent: Agent, args, output_dir: str):
    test_dir = os.path.join(output_dir, f"test_{test_idx:03d}")
    os.makedirs(test_dir, exist_ok=True)
    result_fp = os.path.join(test_dir, "result.json")
    step_log_fp = os.path.join(test_dir, "step_success_log.jsonl")

    logging.info("========== Start test %s ==========", test_idx)
    start_time = time.time()

    obs, _route_obs, infos = env_manager.reset({})
    env_dones = [False] * args.env_num
    overall_success = np.zeros(args.env_num, dtype=bool)
    task_success_cnt = defaultdict(int)
    task_total_cnt = defaultdict(int)
    task_token_usage = defaultdict(empty_token_usage)
    env_token_usage = [empty_token_usage() for _ in range(args.env_num)]
    per_env_results = {}
    trajectories = {str(i): [] for i in range(args.env_num)} if args.record_trajectories else None
    latest_infos = infos

    def write_step_log(step_idx: int, phase: str):
        finished = int(np.array(env_dones).sum().item())
        successes = int(overall_success.sum().item())
        payload = {
            "test_idx": test_idx,
            "step": step_idx,
            "phase": phase,
            "finished_envs": finished,
            "successes": successes,
            "total_envs": args.env_num,
            "success_rate": float(overall_success.mean().item()),
        }
        with open(step_log_fp, "a", encoding="utf-8") as step_f:
            step_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        logging.info(
            "[StepSuccess] test=%s step=%s phase=%s finished=%s/%s successes=%s sr=%.4f",
            test_idx,
            step_idx,
            phase,
            finished,
            args.env_num,
            successes,
            payload["success_rate"],
        )

    for step_idx in range(args.max_steps):
        done_count = int(np.array(env_dones).sum().item())
        current_sr = float(overall_success.mean().item())
        logging.info(
            "Test %s step %s; Dones (%s/%s); SR %.4f",
            test_idx,
            step_idx,
            done_count,
            args.env_num,
            current_sr,
        )
        write_step_log(step_idx, "before_step")

        actions, step_token_usages = infer_actions_concurrently(
            agent=agent,
            text_obs=obs["text"],
            env_dones=env_dones,
            max_workers=args.max_concurrency,
        )
        for i, usage in enumerate(step_token_usages):
            if not env_dones[i]:
                add_token_usage(env_token_usage[i], usage)

        obs_anchor = obs.get("anchor")
        obs, _route_obs, rewards, dones, infos = env_manager.step(
            actions,
            models=[agent.model_name] * len(actions),
        )
        latest_infos = infos

        for i in range(args.env_num):
            if env_dones[i]:
                continue

            if trajectories is not None:
                if isinstance(obs_anchor, list):
                    observation = obs_anchor[i]
                elif obs_anchor is None:
                    observation = None
                else:
                    observation = str(obs_anchor[i])
                trajectories[str(i)].append(
                    {
                        "obs": observation,
                        "output": actions[i],
                        "token_usage": step_token_usages[i],
                    }
                )

            if dones[i]:
                env_dones[i] = True
                won = bool(infos[i].get("won", False))
                overall_success[i] = won
                gamefile = infos[i].get("extra.gamefile", "") or ""
                task = task_from_gamefile(gamefile)
                task_total_cnt[task] += 1
                add_token_usage(task_token_usage[task], env_token_usage[i])
                if won:
                    task_success_cnt[task] += 1
                per_env_results[str(i)] = {
                    "won": won,
                    "task": task,
                    "gamefile": gamefile,
                    "finished_step": step_idx,
                    "token_usage": env_token_usage[i],
                }

        write_step_log(step_idx, "after_step")

        if all(env_dones):
            logging.info("Test %s finished early at step %s", test_idx, step_idx)
            break

    for i in range(args.env_num):
        if env_dones[i]:
            continue

        info = latest_infos[i] if i < len(latest_infos) else {}
        gamefile = info.get("extra.gamefile", "") or ""
        task = task_from_gamefile(gamefile)
        task_total_cnt[task] += 1
        add_token_usage(task_token_usage[task], env_token_usage[i])
        per_env_results[str(i)] = {
            "won": False,
            "task": task,
            "gamefile": gamefile,
            "finished_step": None,
            "token_usage": env_token_usage[i],
        }

    elapsed = time.time() - start_time
    task_rates = {}
    for task in TASKS + ["other"]:
        total = task_total_cnt.get(task, 0)
        if total > 0:
            token_usage = dict(task_token_usage[task])
            task_rates[task] = {
                "success_rate": task_success_cnt[task] / total,
                "success": task_success_cnt[task],
                "total": total,
                "token_usage": token_usage,
                "avg_total_tokens": token_usage["total_tokens"] / total,
                "avg_input_token_cost": token_usage["input_token_cost"] / total,
                "avg_output_token_cost": token_usage["output_token_cost"] / total,
                "avg_token_cost": token_usage["token_cost"] / total,
                "avg_estimated_api_cost": token_usage["estimated_api_cost"] / total,
            }

    total_token_usage = empty_token_usage()
    for usage in env_token_usage:
        add_token_usage(total_token_usage, usage)

    result = {
        "test_idx": test_idx,
        "env_num": args.env_num,
        "max_steps": args.max_steps,
        "finished_envs": int(np.array(env_dones).sum().item()),
        "overall_success_rate": float(overall_success.mean().item()),
        "task_rates": task_rates,
        "token_usage": total_token_usage,
        "elapsed_seconds": elapsed,
        "per_env_results": per_env_results,
        "step_success_log_file": step_log_fp,
    }

    if trajectories is not None:
        trajectories_fp = os.path.join(test_dir, "trajectories.jsonl")
        with open(trajectories_fp, "w", encoding="utf-8") as traj_f:
            for env_idx, steps in trajectories.items():
                record = {
                    "test_idx": test_idx,
                    "env_idx": int(env_idx),
                    "result": per_env_results.get(env_idx),
                    "trajectory": steps,
                }
                traj_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        result["trajectories_file"] = trajectories_fp

        candidate_envs = [env_idx for env_idx, steps in trajectories.items() if steps]
        if candidate_envs:
            rng = random.Random(int(args.seed) + int(test_idx))
            env_idx = rng.choice(candidate_envs)
            random_trajectory_fp = os.path.join(test_dir, "random_trajectory.json")
            random_trajectory = {
                "test_idx": test_idx,
                "env_idx": int(env_idx),
                "result": per_env_results.get(env_idx),
                "trajectory": trajectories[env_idx],
            }
            with open(random_trajectory_fp, "w", encoding="utf-8") as random_f:
                json.dump(random_trajectory, random_f, ensure_ascii=False, indent=2)
            result["random_trajectory_file"] = random_trajectory_fp
            logging.info("Random trajectory file: %s", random_trajectory_fp)

    with open(result_fp, "w", encoding="utf-8") as result_f:
        json.dump(result, result_f, ensure_ascii=False, indent=2)

    logging.info(
        "Test %s overall success: %.4f; elapsed %.2fs; result file: %s",
        test_idx,
        result["overall_success_rate"],
        elapsed,
        result_fp,
    )
    logging.info("[TaskSuccess] test=%s overall success_rate=%.4f total=%s", test_idx, result["overall_success_rate"], args.env_num)
    for task, metrics in sorted(task_rates.items()):
        logging.info(
            "[TaskSuccess] test=%s task=%s success_rate=%.4f success=%s total=%s",
            test_idx,
            task,
            metrics["success_rate"],
            metrics["success"],
            metrics["total"],
        )
    return result


def write_final_summary(results, output_dir: str, model_name: str, log_file: str):
    summary_fp = os.path.join(output_dir, "final_summary.json")

    overall_rates = [r["overall_success_rate"] for r in results]
    task_history = defaultdict(list)
    task_token_history = defaultdict(empty_token_usage)
    task_total_history = defaultdict(int)
    total_token_usage = empty_token_usage()
    for result in results:
        add_token_usage(total_token_usage, result.get("token_usage", {}))
        for task, metrics in result["task_rates"].items():
            task_history[task].append(metrics["success_rate"])
            task_total_history[task] += int(metrics.get("total", 0))
            add_token_usage(task_token_history[task], metrics.get("token_usage", {}))

    summary = {
        "model": model_name,
        "log_file": log_file,
        "num_tests": len(results),
        "overall_success_avg": float(np.mean(overall_rates)) if overall_rates else 0.0,
        "overall_success_std": float(np.std(overall_rates)) if overall_rates else 0.0,
        "token_usage": total_token_usage,
        "tests": [
            {
                "test_idx": r["test_idx"],
                "overall_success_rate": r["overall_success_rate"],
                "finished_envs": r["finished_envs"],
                "elapsed_seconds": r["elapsed_seconds"],
                "token_usage": r.get("token_usage", empty_token_usage()),
            }
            for r in results
        ],
        "task_summary": {
            task: {
                "success_avg": float(np.mean(values)),
                "success_std": float(np.std(values)),
                "num_tests_with_task": len(values),
                "total": int(task_total_history[task]),
                "token_usage": dict(task_token_history[task]),
                "avg_total_tokens": (
                    task_token_history[task]["total_tokens"] / task_total_history[task]
                    if task_total_history[task]
                    else 0.0
                ),
                "avg_input_token_cost": (
                    task_token_history[task]["input_token_cost"] / task_total_history[task]
                    if task_total_history[task]
                    else 0.0
                ),
                "avg_output_token_cost": (
                    task_token_history[task]["output_token_cost"] / task_total_history[task]
                    if task_total_history[task]
                    else 0.0
                ),
                "avg_token_cost": (
                    task_token_history[task]["token_cost"] / task_total_history[task]
                    if task_total_history[task]
                    else 0.0
                ),
                "avg_estimated_api_cost": (
                    task_token_history[task]["estimated_api_cost"] / task_total_history[task]
                    if task_total_history[task]
                    else 0.0
                ),
            }
            for task, values in task_history.items()
        },
    }

    with open(summary_fp, "w", encoding="utf-8") as summary_f:
        json.dump(summary, summary_f, ensure_ascii=False, indent=2)

    lines = [
        "=============== Final Summary ===============",
        f"Model: {model_name}",
        f"Total tests: {summary['num_tests']}",
        (
            "Overall success avg +/- std: "
            f"{summary['overall_success_avg']:.4f} +/- {summary['overall_success_std']:.4f}"
        ),
    ]
    for task, metrics in sorted(summary["task_summary"].items()):
        lines.append(
            f"{task:<35s}: {metrics['success_avg']:.4f} +/- "
            f"{metrics['success_std']:.4f} ({metrics['num_tests_with_task']} tests) | "
            f"tokens={metrics['token_usage']['total_tokens']} "
            f"input_cost={metrics['token_usage']['input_token_cost']:.2f} "
            f"output_cost={metrics['token_usage']['output_token_cost']:.2f} "
            f"token_cost={metrics['token_usage']['token_cost']:.2f}"
        )

    for line in lines:
        logging.info(line)
    logging.info("Final summary file: %s", summary_fp)


def _summarize_values(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
        "values": values,
    }


def aggregate_final_summaries(aggregate_root: Path, json_out: Path) -> dict:
    summaries = []
    for path in sorted(aggregate_root.glob("seed_*/final_summary.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            summaries.append({"path": str(path), "error": str(exc), "_invalid": True})
            continue
        payload["_path"] = str(path)
        summaries.append(payload)

    valid = [s for s in summaries if not s.get("_invalid")]
    model_name = valid[0].get("model", aggregate_root.name) if valid else aggregate_root.name
    overall_values = [float(s.get("overall_success_avg", 0.0)) for s in valid]

    task_values = defaultdict(list)
    task_totals = defaultdict(int)
    for summary in valid:
        for task, metrics in (summary.get("task_summary") or {}).items():
            task_values[task].append(float(metrics.get("success_avg", 0.0)))
            task_totals[task] += int(metrics.get("total", 0))

    result = {
        model_name: {
            "num_runs": len(valid),
            "runs": [s.get("_path") for s in valid],
            "overall_success_rate": _summarize_values(overall_values) if overall_values else None,
            "alfworld_tasks": {
                task: {
                    "success_rate": _summarize_values(values),
                    "total": task_totals[task],
                }
                for task, values in sorted(task_values.items())
            },
        }
    }
    invalid = [s for s in summaries if s.get("_invalid")]
    if invalid:
        result["_skipped"] = invalid

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {json_out}")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-root", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--model",
        "--model-name",
        dest="model",
        default=os.environ.get("FIXED_EVAL_MODEL") or os.environ.get("OPENAI_MODEL_NAME", "deepseek"),
    )
    parser.add_argument(
        "--api-base",
        default=(
            os.environ.get("FIXED_EVAL_API_BASE")
            or os.environ.get("OPENAI_API_BASE")
            or os.environ.get("OPENAI_BASE_URL")
        ),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("FIXED_EVAL_API_KEY") or os.environ.get("OPENAI_API_KEY"),
    )
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("FIXED_EVAL_TEMPERATURE", 0.4)))
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("FIXED_EVAL_TIMEOUT", 120)))
    parser.add_argument("--max-retries", type=int, default=int(os.environ.get("FIXED_EVAL_MAX_RETRIES", 3)))
    parser.add_argument(
        "--max-concurrency",
        "--max-workers",
        dest="max_concurrency",
        type=int,
        default=int(os.environ.get("FIXED_EVAL_MAX_WORKERS") or os.environ.get("MAX_CONCURRENCY", 32)),
    )
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("FIXED_EVAL_MAX_STEPS", 50)))
    parser.add_argument(
        "--env-num",
        type=int,
        default=int(os.environ.get("FIXED_EVAL_ENV_NUM") or os.environ.get("VAL_DATA_SIZE", 200)),
    )
    parser.add_argument("--test-times", type=int, default=int(os.environ.get("FIXED_EVAL_TEST_TIMES", 3)))
    parser.add_argument("--env-name", default=os.environ.get("FIXED_EVAL_ENV_NAME", "alfworld"))
    parser.add_argument("--eval-dataset", default=os.environ.get("FIXED_EVAL_DATASET", "eval_in_distribution"))
    parser.add_argument("--history-length", type=int, default=int(os.environ.get("FIXED_EVAL_HISTORY_LENGTH", 2)))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("FIXED_EVAL_SEED", 1)))
    parser.add_argument(
        "--skills-json-path",
        default=os.environ.get("FIXED_ALFWORLD_SKILLS_JSON_PATH"),
    )
    parser.add_argument(
        "--disable-skills",
        action="store_true",
        default=os.environ.get("FIXED_EVAL_DISABLE_SKILLS", "0").lower() in ("1", "true", "yes"),
    )
    parser.add_argument(
        "--skill-retrieval-service-url",
        default=os.environ.get("FIXED_EVAL_SKILL_RETRIEVAL_SERVICE_URL"),
    )
    parser.add_argument("--retrieval-mode", default=os.environ.get("FIXED_EVAL_RETRIEVAL_MODE", "embedding"))
    parser.add_argument(
        "--embedding-model-path",
        default=os.environ.get("FIXED_EVAL_EMBEDDING_MODEL_PATH"),
    )
    parser.add_argument(
        "--skill-text-for-retrieval",
        default=os.environ.get("FIXED_EVAL_SKILL_TEXT_FOR_RETRIEVAL", "when_to_apply"),
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=float(os.environ.get("FIXED_EVAL_SIMILARITY_THRESHOLD", 0.7)),
    )
    parser.add_argument("--top-k-task", type=int, default=int(os.environ.get("FIXED_EVAL_TOP_K_TASK", 3)))
    parser.add_argument("--top-k-step", type=int, default=int(os.environ.get("FIXED_EVAL_TOP_K_STEP", 3)))
    parser.add_argument("--skill-gen-mode", default=os.environ.get("FIXED_EVAL_SKILL_GEN_MODE", "task_step"))
    parser.add_argument(
        "--output-root",
        default=os.environ.get(
            "FIXED_EVAL_OUTPUT_ROOT",
            str(PROJECT_ROOT / "outputs" / "fixed_model_alfworld_eval"),
        ),
    )
    parser.add_argument("--output-dir", default=os.environ.get("FIXED_EVAL_OUTPUT_DIR"))
    parser.add_argument("--record-trajectories", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.aggregate_root is not None:
        json_out = args.json_out or args.aggregate_root / "fixed_route_metric_summary.json"
        aggregate_final_summaries(args.aggregate_root, json_out)
        return

    model_name, api_base, api_key = resolve_model(args.model, args.api_base, args.api_key)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(args.output_root, f"{model_name}_{run_id}")
    log_file = setup_logger(output_dir)

    logging.info("Output directory: %s", output_dir)
    logging.info(
        "Model=%s env_num=%s test_times=%s max_steps=%s max_concurrency=%s history_length=%s",
        model_name,
        args.env_num,
        args.test_times,
        args.max_steps,
        args.max_concurrency,
        args.history_length,
    )
    logging.info(
        "Skills enabled=%s skills_json_path=%s retrieval_mode=%s top_k_task=%s top_k_step=%s skill_retrieval_service_url=%s",
        not args.disable_skills,
        args.skills_json_path,
        args.retrieval_mode,
        args.top_k_task,
        args.top_k_step,
        args.skill_retrieval_service_url,
    )

    env_manager = build_env(
        env_name=args.env_name,
        env_num=args.env_num,
        seed=args.seed,
        max_steps=args.max_steps,
        history_length=args.history_length,
        eval_dataset=args.eval_dataset,
        use_skills_only_memory=not args.disable_skills,
        skills_json_path=args.skills_json_path,
        skill_retrieval_service_url=args.skill_retrieval_service_url,
        retrieval_mode=args.retrieval_mode,
        embedding_model_path=args.embedding_model_path,
        skill_text_for_retrieval=args.skill_text_for_retrieval,
        similarity_threshold=args.similarity_threshold,
        top_k_task=args.top_k_task,
        top_k_step=args.top_k_step,
        skill_gen_mode=args.skill_gen_mode,
    )
    agent = Agent(
        model_name=model_name,
        api_key=api_key,
        api_base=api_base,
        temperature=args.temperature,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )

    results = []
    try:
        for test_idx in range(args.test_times):
            results.append(run_one_test(test_idx, env_manager, agent, args, output_dir))
    finally:
        if hasattr(env_manager, "envs") and hasattr(env_manager.envs, "close"):
            env_manager.envs.close()

    write_final_summary(results, output_dir, model_name, log_file)


if __name__ == "__main__":
    main()
