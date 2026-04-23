# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl import DataProto
import torch
import numpy as np
from collections import deque

class EpisodeRewardManager:
    """The reward manager.
    """

    def __init__(
        self,
        tokenizer,
        num_examine,
        normalize_by_length=False,
        cost_coe=0.0,
        cost_apply_on_nonpositive=False,
        cost_normalization_window=1000,
        cost_percentile_low=0.05,
        cost_percentile_high=0.95,
        cost_transform="sqrt",
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.normalize_by_length = normalize_by_length
        self.cost_coe = float(cost_coe or 0.0)
        self.cost_apply_on_nonpositive = bool(cost_apply_on_nonpositive)
        self.cost_percentile_low = float(cost_percentile_low)
        self.cost_percentile_high = float(cost_percentile_high)
        self.cost_transform = cost_transform
        self._cost_eps = 1e-8
        self._cost_buffer = deque(maxlen=int(cost_normalization_window or 1000))

    def _preprocess_cost(self, cost: float) -> float:
        cost = max(float(cost), 0.0)
        if self.cost_transform == "log":
            return float(np.log1p(0.01 * cost))
        if self.cost_transform == "sqrt":
            return float(np.sqrt(cost))
        return cost

    def _normalize_cost_reward(self, cost: float) -> float:
        """Return a Router-R1-style reward where lower routing cost is better."""
        processed = self._preprocess_cost(cost)
        self._cost_buffer.append(processed)
        arr = np.asarray(self._cost_buffer, dtype=np.float64)
        if arr.size >= 2:
            cost_min = np.percentile(arr, 100 * self.cost_percentile_low)
            cost_max = np.percentile(arr, 100 * self.cost_percentile_high)
        else:
            cost_min = arr.min()
            cost_max = arr.max()

        denom = cost_max - cost_min
        if denom < self._cost_eps:
            return 0.5

        scaled = (processed - cost_min) / denom
        return 1.0 - float(np.clip(scaled, 0.0, 1.0))

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = {
            "base_episode_rewards": [],
            "api_costs": [],
            "cost_rewards": [],
            "final_rewards": [],
        }

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)

            # ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)
            multi_modal_inputs = data_item.non_tensor_batch.get('multi_modal_inputs', None)
            if multi_modal_inputs is not None:
                pixel_values = multi_modal_inputs['pixel_values']
                image_grid_thw = multi_modal_inputs['image_grid_thw']


            episode_rewards = data_item.non_tensor_batch['episode_rewards']
            episode_lengths = data_item.non_tensor_batch['episode_lengths']

            if self.normalize_by_length:
                base_score = episode_rewards / episode_lengths
            else:
                base_score = episode_rewards
            api_cost = float(data_item.non_tensor_batch.get('api_costs', 0.0))
            cost_reward = self._normalize_cost_reward(api_cost)
            if self.cost_coe > 0 and (self.cost_apply_on_nonpositive or float(base_score) > 0):
                score = float(base_score) * (1.0 - self.cost_coe) + cost_reward * self.cost_coe
            else:
                score = float(base_score)
            reward_tensor[i, valid_response_length - 1] = torch.tensor(score, dtype=torch.float32, device=prompt_ids.device)
            reward_extra_info["base_episode_rewards"].append(float(base_score))
            reward_extra_info["api_costs"].append(api_cost)
            reward_extra_info["cost_rewards"].append(float(cost_reward))
            reward_extra_info["final_rewards"].append(float(score))

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine and np.random.random() < 0.1:
                already_print_data_sources[data_source] += 1
                print(f"[{data_source}][prompt]", prompt_str)
                print(f"[{data_source}][response]", response_str)
                print(f"[{data_source}][base_score]", base_score)
                print(f"[{data_source}][api_cost]", api_cost)
                print(f"[{data_source}][cost_reward]", cost_reward)
                print(f"[{data_source}][score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
