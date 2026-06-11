from __future__ import annotations

import concurrent.futures
from typing import Any, Dict, List

import numpy as np
import requests


class TextCraftHTTPClient:
    def __init__(
        self,
        env_addr: str,
        timeout: int = 600,
        minecraft_dir: str = "agentenv_textcraft/",
        commands: Any = None,
        goal: Any = None,
    ):
        self.env_addr = env_addr.rstrip("/")
        self.timeout = timeout
        response = self._post(
            "create",
            {
                "minecraft_dir": minecraft_dir,
                "commands": commands,
                "goal": goal,
            },
        )
        self.env_id = int(response["id"])

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        res = requests.post(f"{self.env_addr}/{path}", json=payload, timeout=self.timeout)
        res.raise_for_status()
        data = res.json()
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(data["error"])
        return data

    def reset(self, data_idx: int):
        return self._post("reset", {"id": self.env_id, "data_idx": int(data_idx)})

    def step(self, action: str):
        return self._post("step", {"id": self.env_id, "action": action})

    def close(self):
        try:
            self._post("close", {"id": self.env_id})
        except Exception:
            pass


class TextCraftMultiProcessEnv:
    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        is_train: bool = True,
        env_kwargs: dict | None = None,
    ) -> None:
        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        if not is_train:
            assert group_n == 1

        env_kwargs = env_kwargs or {}
        self.env_addr = env_kwargs.get("env_addr", "http://127.0.0.1:36005")
        self.timeout = int(env_kwargs.get("timeout", 600))
        self.minecraft_dir = env_kwargs.get("minecraft_dir", "agentenv_textcraft/")
        self.commands = env_kwargs.get("commands")
        self.goal = env_kwargs.get("goal")
        self.data_len = int(env_kwargs.get("data_len", 374))
        self.val_offset = int(env_kwargs.get("val_offset", 10000))
        self._rng = np.random.RandomState(seed)
        self._reset_count = 0
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=min(self.num_processes, 256))
        self._clients = [
            TextCraftHTTPClient(
                self.env_addr,
                timeout=self.timeout,
                minecraft_dir=self.minecraft_dir,
                commands=self.commands,
                goal=self.goal,
            )
            for _ in range(self.num_processes)
        ]
        self._dones = [False] * self.num_processes
        self._last_obs = [""] * self.num_processes
        self._active_count = self.num_processes

    def _sample_indices(self) -> List[int]:
        if self.is_train:
            base = self._rng.choice(self.data_len, size=self.env_num, replace=self.env_num > self.data_len)
        else:
            start = self.val_offset + self._reset_count * self.env_num
            base = np.arange(start, start + self.env_num) % self.data_len
        self._reset_count += 1
        return np.repeat(base, self.group_n).astype(int).tolist()

    def _indices_from_kwargs(self, kwargs) -> List[int] | None:
        if kwargs is None:
            return None
        values = kwargs
        if isinstance(values, dict):
            values = values.get("data_idx", values.get("item_id"))
        if values is None:
            return None
        indices = []
        for value in list(values):
            if isinstance(value, dict):
                value = value.get("data_idx", value.get("item_id"))
            indices.append(int(value))
        if len(indices) > self.num_processes:
            raise ValueError(f"Expected at most {self.num_processes} TextCraft data_idx values, got {len(indices)}")
        return indices

    def _metadata_from_kwargs(self, kwargs) -> List[Dict[str, Any]]:
        if kwargs is None:
            return []
        values = kwargs
        if isinstance(values, dict):
            data_values = values.get("data_idx", values.get("item_id"))
            depth_values = values.get("depth")
            if data_values is None:
                return []
            if not isinstance(data_values, (list, tuple, np.ndarray)):
                data_values = [data_values]
            if depth_values is not None and not isinstance(depth_values, (list, tuple, np.ndarray)):
                depth_values = [depth_values] * len(data_values)
            metadata = []
            for i, data_idx in enumerate(list(data_values)):
                item = {"data_idx": int(data_idx)}
                if depth_values is not None and i < len(depth_values):
                    item["depth"] = depth_values[i]
                metadata.append(item)
            return metadata
        metadata = []
        for value in list(values):
            if isinstance(value, dict):
                item = {}
                data_idx = value.get("data_idx", value.get("item_id"))
                if data_idx is not None:
                    item["data_idx"] = int(data_idx)
                if "depth" in value:
                    item["depth"] = value["depth"]
                metadata.append(item)
            else:
                metadata.append({"data_idx": int(value)})
        return metadata

    def reset(self, kwargs=None):
        indices = self._indices_from_kwargs(kwargs) or self._sample_indices()
        metadata = self._metadata_from_kwargs(kwargs)
        self._active_count = len(indices)
        futures = [self._executor.submit(client.reset, idx) for client, idx in zip(self._clients, indices)]
        obs_list, info_list = [], []
        for env_i, (idx, future) in enumerate(zip(indices, futures)):
            payload = future.result()
            obs_list.append(payload["observation"])
            self._dones[env_i] = False
            self._last_obs[env_i] = payload["observation"]
            info = {
                "won": False,
                "data_idx": idx,
                "goal": _extract_goal(payload["observation"]),
                "task_score": 0.0,
            }
            if env_i < len(metadata) and "depth" in metadata[env_i]:
                info["depth"] = metadata[env_i]["depth"]
            info_list.append(info)
        return obs_list, info_list

    def step(self, actions: List[str]):
        expected = int(getattr(self, "_active_count", self.num_processes))
        if len(actions) != expected:
            raise ValueError(f"Expected {expected} actions, got {len(actions)}")
        futures = [
            None if self._dones[i] else self._executor.submit(client.step, action)
            for i, (client, action) in enumerate(zip(self._clients[:expected], actions))
        ]
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for i, future in enumerate(futures):
            if future is None:
                obs_list.append(self._last_obs[i])
                reward_list.append(0.0)
                done_list.append(True)
                info_list.append({"won": False, "task_score": 0.0})
                continue
            payload = future.result()
            reward = float(payload.get("reward", 0.0) or 0.0)
            done = bool(payload.get("done", False))
            won = bool(done and reward > 0)
            obs = payload.get("observation", "")
            self._last_obs[i] = obs
            self._dones[i] = done
            obs_list.append(obs)
            reward_list.append(10.0 if won else 0.0)
            done_list.append(done)
            info_list.append({"won": won, "task_score": reward})
        return obs_list, reward_list, done_list, info_list

    def close(self):
        for client in getattr(self, "_clients", []):
            client.close()
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)


def _extract_goal(observation: str) -> str:
    marker = "Goal:"
    if marker not in observation:
        return ""
    return observation.split(marker, 1)[1].strip()


def build_textcraft_envs(
    seed: int,
    env_num: int,
    group_n: int,
    is_train: bool = True,
    env_kwargs: dict | None = None,
    resources_per_worker: dict | None = None,
):
    return TextCraftMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        is_train=is_train,
        env_kwargs=env_kwargs,
    )
