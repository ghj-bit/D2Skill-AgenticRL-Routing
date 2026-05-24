#!/usr/bin/env python3
"""Direct fixed-model ALFWorld eval aligned with prompt_agent/gpt4o_alfworld.py."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILLRL_ENV_PATH = PROJECT_ROOT / "SkillRL" / "agent_system" / "environments"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import agent_system.environments as environments_pkg  # noqa: E402

if SKILLRL_ENV_PATH.exists() and str(SKILLRL_ENV_PATH) not in environments_pkg.__path__:
    environments_pkg.__path__.append(str(SKILLRL_ENV_PATH))

from agent_system.environments.env_package.alfworld import (  # noqa: E402
    alfworld_projection,
    build_alfworld_envs,
)
from agent_system.memory import SimpleMemory  # noqa: E402
from routing.models_config.models_config import MODEL_CONF  # noqa: E402


ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

TASKS = [
    "pick_and_place",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
]


def to_numpy(data: Any) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data
    return np.array(data)


def parse_gamefile(infos: List[Dict[str, Any]]) -> List[Any]:
    gamefile = []
    for info in infos:
        if "extra.gamefile" in info:
            gamefile.append(info["extra.gamefile"])
        else:
            gamefile.append(None)
    return gamefile


def set_gamefile(infos: List[Dict[str, Any]], gamefile: List[Any]) -> List[Dict[str, Any]]:
    for i in range(len(infos)):
        if "extra.gamefile" in infos[i]:
            infos[i]["extra.gamefile"] = gamefile[i]
        else:
            infos[i]["extra.gamefile"] = None
    return infos


class PromptAgentAlfWorldManager:
    """Compatibility manager matching verl-agent prompt_agent AlfWorld prompts."""

    def __init__(self, envs, projection_f, history_length: int):
        self.envs = envs
        self.projection_f = projection_f
        self.config = SimpleNamespace(env=SimpleNamespace(history_length=history_length))
        self.memory = SimpleMemory()

    def reset(self, kwargs):
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        self.memory.reset(batch_size=len(text_obs))
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)
        return {"text": full_text_obs, "image": image_obs, "anchor": text_obs}, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({"text_obs": self.pre_text_obs, "action": actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        next_observations = {"text": full_text_obs, "image": image_obs, "anchor": text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)
        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find("Your task is to: ")
            if task_start != -1:
                self.tasks.append(obs[task_start + len("Your task is to: "):].strip())
            else:
                raise ValueError("Task description not found in text observation.")

    def build_text_obs(
        self,
        text_obs: List[str],
        admissible_actions: List[List[str]],
        init: bool = False,
    ) -> List[str]:
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                self.config.env.history_length,
                obs_key="text_obs",
                action_key="action",
            )

        for i in range(len(text_obs)):
            reformatted_admissible_actions = "\n ".join(
                f"'{s}'" for s in admissible_actions[i] if s != "help"
            )

            if init or self.config.env.history_length <= 0:
                obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions,
                )
            else:
                obs = ALFWORLD_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions,
                )

            postprocess_text_obs.append(obs)
        return postprocess_text_obs


def resolve_model(model: str, api_base: str | None, api_key: str | None) -> Tuple[str, str, str]:
    normalized = str(model or "").strip().lower()
    if "qwen3-30b" in normalized or normalized in {"qwen-30b", "qwen30b"}:
        resolved_model = "Qwen3-30B"
    elif "qwen3" in normalized or normalized in {"qwen", "qwen-8b", "qwen8b"}:
        resolved_model = "qwen3-8B"
    elif "deepseek" in normalized:
        resolved_model = "deepseek"
    else:
        resolved_model = model

    model_conf = MODEL_CONF.get(resolved_model, {})
    base_url = api_base or model_conf.get("api_base") or os.environ.get("OPENAI_BASE_URL", "")
    key = api_key or model_conf.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
    if not base_url:
        raise ValueError(f"No api_base found for model {model!r}.")
    if not key:
        raise ValueError(f"No api_key found for model {model!r}.")
    return resolved_model, base_url, key


class Agent:
    def __init__(self, model_name: str, api_base: str, api_key: str):
        self.model_name = model_name
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The fixed-model eval script requires the `openai` package."
            ) from exc
        self.client = OpenAI(
            base_url=api_base,
            api_key=api_key,
        )

    def get_action_from_gpt(self, obs: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": obs,
                }
            ],
            temperature=0.4,
            n=1,
            stop=None,
        )
        action = response.choices[0].message.content.strip()
        return action


def build_env(env_name: str, env_num: int = 1, seed: int = 1, eval_dataset: str = "eval_in_distribution", history_length: int = 4):
    group_n = 1
    if env_name == "alfworld":
        alf_config_path = SKILLRL_ENV_PATH / "env_package" / "alfworld" / "configs" / "config_tw.yaml"
        if not alf_config_path.exists():
            alf_config_path = PROJECT_ROOT / "agent_system" / "environments" / "env_package" / "alfworld" / "configs" / "config_tw.yaml"
        env_kwargs = {
            "eval_dataset": eval_dataset,
        }
        resources_per_worker = {"num_cpus": 0.05, "num_gpus": 0.0}
        envs = build_alfworld_envs(
            str(alf_config_path),
            seed=seed,
            env_num=env_num,
            group_n=group_n,
            is_train=False,
            env_kwargs=env_kwargs,
            resources_per_worker=resources_per_worker,
        )
        env_manager = PromptAgentAlfWorldManager(envs, alfworld_projection, history_length)
    else:
        raise ValueError(f"Unsupported environment name: {env_name}")

    return env_manager


def mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return float(math.sqrt(sum((x - m) ** 2 for x in values) / len(values)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("FIXED_EVAL_MODEL", "deepseek"))
    parser.add_argument("--api-base", default=os.environ.get("FIXED_EVAL_API_BASE"))
    parser.add_argument("--api-key", default=os.environ.get("FIXED_EVAL_API_KEY"))
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("FIXED_EVAL_MAX_STEPS", 50)))
    parser.add_argument("--env-num", type=int, default=int(os.environ.get("FIXED_EVAL_ENV_NUM", 200)))
    parser.add_argument("--test-times", type=int, default=int(os.environ.get("FIXED_EVAL_TEST_TIMES", 3)))
    parser.add_argument("--env-name", default=os.environ.get("FIXED_EVAL_ENV_NAME", "alfworld"))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("FIXED_EVAL_SEED", 1)))
    parser.add_argument("--eval-dataset", default=os.environ.get("FIXED_EVAL_DATASET", "eval_in_distribution"))
    parser.add_argument("--history-length", type=int, default=int(os.environ.get("FIXED_EVAL_HISTORY_LENGTH", 4)))
    parser.add_argument("--max-workers", type=int, default=int(os.environ.get("FIXED_EVAL_MAX_WORKERS", 32)))
    parser.add_argument("--output-dir", default=os.environ.get("FIXED_EVAL_OUTPUT_DIR"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_name, api_base, api_key = resolve_model(args.model, args.api_base, args.api_key)

    os.makedirs("logs/alfworld", exist_ok=True)
    log_fp = os.path.join(
        "logs/alfworld",
        f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler(log_fp, encoding="utf-8"), logging.StreamHandler()],
    )

    env_manager = build_env(
        args.env_name,
        args.env_num,
        seed=args.seed,
        eval_dataset=args.eval_dataset,
        history_length=args.history_length,
    )
    agent = Agent(model_name=model_name, api_base=api_base, api_key=api_key)

    output_dir = Path(
        args.output_dir
        or PROJECT_ROOT / "outputs" / "fixed_model_alfworld_eval" / f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_success_rates = []
    task_success_history = defaultdict(list)
    rounds = []
    total_start_time = time.time()

    try:
        with tqdm(total=args.test_times * args.max_steps, desc="fixed-model eval", unit="step") as pbar:
            for test_idx in range(args.test_times):
                logging.info(f"\n========== Start test {test_idx} ==========")
                start_time = time.time()

                kwargs = {}
                obs, infos = env_manager.reset(kwargs)
                env_dones = [False] * args.env_num

                overall_success_this_round = np.zeros(args.env_num, dtype=bool)
                task_success_cnt = defaultdict(int)
                task_total_cnt = defaultdict(int)

                for step_idx in range(args.max_steps):
                    logging.info(
                        f"Step {step_idx}; Dones ({np.array(env_dones).sum().item()}/{args.env_num}); "
                        f"SR {overall_success_this_round.mean().item()}"
                    )

                    actions = ["None"] * args.env_num
                    pending = [i for i in range(args.env_num) if not env_dones[i]]
                    prompts = [obs["text"][i] for i in pending]
                    if args.max_workers <= 1:
                        generated = [agent.get_action_from_gpt(prompt) for prompt in prompts]
                    else:
                        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                            generated = list(executor.map(agent.get_action_from_gpt, prompts))
                    for i, action in zip(pending, generated):
                        actions[i] = action

                    obs, rewards, dones, infos = env_manager.step(actions)

                    for i in range(args.env_num):
                        if env_dones[i]:
                            continue

                        if dones[i]:
                            env_dones[i] = True
                            won = bool(infos[i].get("won", False))
                            overall_success_this_round[i] = won

                            gamefile = infos[i].get("extra.gamefile", "")
                            matched = False
                            for task in TASKS:
                                if task in gamefile:
                                    task_total_cnt[task] += 1
                                    if won:
                                        task_success_cnt[task] += 1
                                    matched = True
                                    break
                            if not matched:
                                task_total_cnt["other"] += 1
                                if won:
                                    task_success_cnt["other"] += 1

                    pbar.set_postfix(
                        test=test_idx,
                        step=step_idx,
                        done=f"{np.array(env_dones).sum().item()}/{args.env_num}",
                        sr=f"{overall_success_this_round.mean().item():.4f}",
                    )
                    pbar.update(1)

                    if all(env_dones):
                        logging.info("All environments finished early!")
                        pbar.update(args.max_steps - step_idx - 1)
                        break

                round_success_rate = overall_success_this_round.mean()
                overall_success_rates.append(round_success_rate)
                logging.info(f"Test {test_idx} overall success: {round_success_rate:.4f}")

                round_tasks = {}
                for task in TASKS + ["other"]:
                    if task_total_cnt.get(task, 0) > 0:
                        rate = task_success_cnt[task] / task_total_cnt[task]
                        task_success_history[task].append(rate)
                        round_tasks[task] = {
                            "success_rate": rate,
                            "success_count": int(task_success_cnt[task]),
                            "count": int(task_total_cnt[task]),
                        }
                        logging.info(
                            f"    {task:<35s}: {rate:.4f} "
                            f"({task_success_cnt[task]}/{task_total_cnt[task]})"
                        )

                elapsed = time.time() - start_time
                logging.info(f"Test {test_idx} time elapsed: {elapsed:.2f}s\n")

                round_summary = {
                    "round": test_idx,
                    "overall_success_rate": float(round_success_rate),
                    "elapsed_seconds": elapsed,
                    "tasks": round_tasks,
                }
                rounds.append(round_summary)
                (output_dir / f"round_{test_idx}_summary.json").write_text(
                    json.dumps(round_summary, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

        logging.info("=============== Final Summary ===============")
        logging.info(
            f"Total tests: {args.test_times} | Envs / test: {args.env_num} | "
            f"Total envs: {args.env_num * args.test_times}"
        )
        logging.info(
            f"Overall success avg +/- std: "
            f"{np.mean(overall_success_rates):.4f} +/- {np.std(overall_success_rates):.4f}"
        )

        final_tasks = {}
        for task in TASKS + ["other"]:
            if task_success_history.get(task):
                task_mean = float(np.mean(task_success_history[task]))
                task_std = float(np.std(task_success_history[task]))
                final_tasks[task] = {
                    "mean_success_rate": task_mean,
                    "std_success_rate": task_std,
                    "round_count": len(task_success_history[task]),
                }
                logging.info(f"{task:<35s}: {task_mean:.4f} +/- {task_std:.4f}")

        result = {
            "model": model_name,
            "args": vars(args),
            "log_file": log_fp,
            "elapsed_seconds": time.time() - total_start_time,
            "aggregate": {
                "overall_mean_success_rate": float(np.mean(overall_success_rates)) if overall_success_rates else 0.0,
                "overall_std_success_rate": float(np.std(overall_success_rates)) if overall_success_rates else 0.0,
                "round_count": len(rounds),
                "tasks": final_tasks,
            },
            "rounds": rounds,
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        logging.info(f"Wrote summary to {summary_path}")
    finally:
        close = getattr(env_manager.envs, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
