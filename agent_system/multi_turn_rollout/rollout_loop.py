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

import torch
import numpy as np
import json
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
import os
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.environments import EnvironmentManagerBase
from typing import List, Dict, Any, Optional
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from routing.llm_agent.route_service import access_routing_pool, check_llm_name
from routing.models_config.models_config import MODEL_CONF
import re
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)

ALFWORLD_TASKS = (
    "pick_and_place",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
)


def _alfworld_task_from_info(info: Dict[str, Any]) -> str:
    gamefile = ""
    if isinstance(info, dict):
        gamefile = info.get("extra.gamefile", "") or ""
    for task in ALFWORLD_TASKS:
        if task in gamefile:
            return task
    return "other" if gamefile else ""

def _skill_input_to_retrieval(s: Dict[str, Any], mode: str = "full") -> str:
    """Text that was used as input (document side) for this skill in retrieval.
    mode: 'full' = title + principle + when_to_apply; 'when_to_apply' = only when_to_apply; 'principle' = only principle.
    """
    if mode == "when_to_apply":
        return (s.get("when_to_apply") or "").strip()
    if mode == "principle":
        return (s.get("principle") or "").strip()
    parts = [s.get("title", ""), s.get("principle", ""), s.get("when_to_apply", "")]
    return ". ".join(p for p in parts if p and str(p).strip()).strip(". ")


def _task_step_skill_row(s: Dict[str, Any]) -> Dict[str, Any]:
    """One row for task_skill or step_skill in snapshot (skill_id, title, input_to_retrieval, similarity, utility, ucb, retrieval_score)."""
    inp = (s.get("retrieval_obs") or "").strip() or _skill_input_to_retrieval(s, "full")
    row = {"title": s.get("title", ""), "input_to_retrieval": inp, "similarity": s.get("similarity")}
    if s.get("skill_id") is not None:
        row["skill_id"] = s["skill_id"]
    if "utility" in s:
        row["utility"] = s["utility"]
    if "ucb" in s:
        row["ucb"] = s["ucb"]
    if "retrieval_score" in s:
        row["retrieval_score"] = s["retrieval_score"]
    return row


def _snapshot_retrieved_memories(mem: Dict[str, Any], skill_text_mode: str = "full") -> Dict[str, Any]:
    """For JSON: query_text and per-skill task_skills, step_skills."""
    return {
        "query_text": mem.get("query_text", ""),
        "task_skills": [_task_step_skill_row(s) for s in mem.get("task_skills", [])],
        "step_skills": [_task_step_skill_row(s) for s in mem.get("step_skills", [])],
    }

class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self._random_trace_dumped = False

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
    ):
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        
        Parameters:
            item (int): Sample index in the batch
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
        
        Returns:
            dict: Contains processed input data such as input_ids, attention_mask, etc.
        """

        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        
        # Get observation components
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_query_texts = obs.get('query_text', None)
        obs_systems = obs.get('system', None)
        obs_chats = obs.get('chat', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        obs_query_text = (obs_query_texts[item] if obs_query_texts is not None and item < len(obs_query_texts) else None) or ""
        obs_system = (obs_systems[item] if obs_systems is not None and item < len(obs_systems) else None) or ""
        obs_chat = obs_chats[item] if obs_chats is not None and item < len(obs_chats) else None
        is_multi_modal = obs_image is not None

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        # Build chat structure
        # obs_content = raw_prompt[0]['content']
        # if '<image>' in obs_content: 
        #     obs_content = obs_content.replace('<image>', '')

        # Build chat structure
        obs_content = ''
        if obs_text is not None:
            obs_content += obs_text
        else:
            print(f"Warning: No text observation found!")

        
        if obs_chat is not None:
            chat_messages = obs_chat
        else:
            chat_messages = []
            if obs_system:
                chat_messages.append({
                    "content": obs_system,
                    "role": "system",
                })
            chat_messages.append({
                "content": obs_content,
                "role": "user",
            })
        chat = np.array(chat_messages)
        
        # Apply chat template
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs
        )
        
        # Initialize return dict
        row_dict = {}
        
        # Process multimodal data
        if is_multi_modal:
            # Replace image placeholder with vision tokens
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)

        else:
            raw_prompt = prompt_with_chat_template
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                            tokenizer=self.tokenizer,
                                                                            max_length=self.config.data.max_prompt_length,
                                                                            pad_token_id=self.tokenizer.pad_token_id,
                                                                            left_pad=True,
                                                                            truncation=self.config.data.truncation,)
        
        

        if is_multi_modal:

            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")

        # Build final output dict
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
            'anchor_obs': _obs_anchor,
            'query_text': obs_query_text,
            'index': item,
            'data_source': data_source
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat.tolist()
        
        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto,
        obs: Dict,
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        
        # Process each sample in parallel
        for item in range(batch_size):
            # Extract per-sample observations
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
            )
            processed_samples.append(processed)
        
        # Aggregate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch

    def keep_router_outputs_only(
        self,
        batch: DataProto,
        router_outputs: List[str],
        routed_outputs: List[str],
    ) -> DataProto:
        """Keep only router outputs in model responses; routed outputs live in non_tensor_batch."""
        responses = batch.batch["responses"]
        attention_mask = batch.batch["attention_mask"]
        input_ids = batch.batch["input_ids"]
        prompt_length = batch.batch["prompts"].shape[-1]
        response_length = responses.shape[-1]
        loss_mask = torch.zeros_like(attention_mask)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        for i in range(responses.shape[0]):
            router_ids = self.tokenizer(
                str(router_outputs[i]) if i < len(router_outputs) else "",
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
            router_len = min(len(router_ids), response_length)

            attention_mask[i, prompt_length:prompt_length + response_length] = 0
            if pad_id is not None:
                responses[i, :] = pad_id
                input_ids[i, prompt_length:prompt_length + response_length] = pad_id

            if router_len <= 0:
                continue

            router_tensor = torch.tensor(router_ids[:router_len], dtype=responses.dtype, device=responses.device)
            responses[i, :router_len] = router_tensor
            input_ids[i, prompt_length:prompt_length + router_len] = router_tensor
            attention_mask[i, prompt_length:prompt_length + router_len] = 1
            loss_mask[i, prompt_length:prompt_length + router_len] = 1

        batch.batch["responses"] = responses
        batch.batch["input_ids"] = input_ids
        batch.batch["attention_mask"] = attention_mask
        batch.batch["position_ids"] = compute_position_id_with_mask(attention_mask)
        batch.batch["loss_mask"] = loss_mask
        return batch

    def _build_forced_router_batch_output(self, batch_input: DataProto) -> DataProto:
        """Create a rollout output without running the router model for fixed-route eval."""
        input_ids = batch_input.batch["input_ids"]
        attention_mask = batch_input.batch["attention_mask"]
        batch_size = input_ids.shape[0]
        response_length = int(self.config.data.max_response_length)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0

        responses = torch.full(
            (batch_size, response_length),
            int(pad_id),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        response_attention_mask = torch.zeros(
            (batch_size, response_length),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        input_with_response = torch.cat([input_ids, responses], dim=-1)
        full_attention_mask = torch.cat([attention_mask, response_attention_mask], dim=-1)
        full_position_ids = compute_position_id_with_mask(full_attention_mask)
        rollout_log_probs = torch.zeros(
            (batch_size, response_length),
            dtype=torch.float32,
            device=input_ids.device,
        )
        return DataProto.from_dict(
            tensors={
                "prompts": input_ids,
                "responses": responses,
                "input_ids": input_with_response,
                "rollout_log_probs": rollout_log_probs,
                "attention_mask": full_attention_mask,
                "position_ids": full_position_ids,
            }
        )


    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            api_costs: Optional[np.ndarray] = None,
            per_step_retrieved: Optional[List[List[Dict]]] = None,
            envs: Optional[Any] = None,
            enable_dynamic_management: bool = False,
            with_skills_per_traj: Optional[np.ndarray] = None,
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
            tool_callings (np.ndarray): Number of tool callings for each environment
            api_costs (np.ndarray): Accumulated external routing API costs for each environment
        Returns:
            DataProto: Collected and organized trajectory data
        """
        batch_size = len(total_batch_list)
        if api_costs is None:
            api_costs = np.zeros(batch_size, dtype=np.float32)
        else:
            api_costs = np.asarray(api_costs, dtype=np.float32).ravel()

        wsm_arr = with_skills_per_traj
        if wsm_arr is None and envs is not None:
            _w = getattr(envs, "with_skills_mask", None)
            wsm_arr = np.asarray(_w, dtype=bool).copy() if _w is not None else None
        if wsm_arr is not None:
            wsm_arr = np.asarray(wsm_arr, dtype=bool).ravel()

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)

        # A/B rollout: split success for skill vs no-skill arms (logged as episode/success_rate_* via metric_utils)
        if (
            wsm_arr is not None
            and "success_rate" in success
            and wsm_arr.shape[0] == batch_size
        ):
            st = np.asarray(success["success_rate"], dtype=np.float64).ravel()
            if st.shape[0] == batch_size:
                skill_vals = st[wsm_arr]
                origin_vals = st[~wsm_arr]
                success_rate["success_rate_skill"] = np.array(
                    [float(np.mean(skill_vals)) if skill_vals.size > 0 else float("nan")],
                    dtype=np.float32,
                )
                success_rate["success_rate_origin"] = np.array(
                    [float(np.mean(origin_vals)) if origin_vals.size > 0 else float("nan")],
                    dtype=np.float32,
                )

        effective_batch = []
        for bs in range(batch_size):
            # sum the rewards for each data in total_batch_list[bs]
            for data in total_batch_list[bs]:
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # episode_rewards
                    data['episode_rewards'] = episode_rewards[bs]
                    # episode_lengths
                    data['episode_lengths'] = episode_lengths[bs]
                    # tool_callings
                    data['tool_callings'] = tool_callings[bs]
                    data['api_costs'] = api_costs[bs]
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value
                    # trajectory index for intrinsic reward / utility (stable after balance_batch)
                    data['traj_index'] = bs

                    effective_batch.append(data)
            
        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        # Per-step retrieval for recording only: always trajectory-level (len = num_trajectories).
        # Trainer will pop this before adjust_batch so it never causes length mismatch.
        if per_step_retrieved is not None:
            # Must store as an object array with shape (n_traj,), where each cell
            # is one trajectory-level list[dict]. If we directly call np.array on
            # list_of_lists and all trajectories share the same step count, NumPy
            # creates a 2D (n_traj, L) matrix. A later .ravel() would flatten by
            # step instead of by trajectory, and JSON records would keep only the
            # global i-th step for sample_i.
            n_ps = len(per_step_retrieved)
            _ps_store = np.empty(n_ps, dtype=object)
            for _ii in range(n_ps):
                _ps_store[_ii] = list(per_step_retrieved[_ii])
            gen_batch_output.non_tensor_batch["per_step_retrieved_for_record"] = _ps_store

        traj_idx = np.asarray(gen_batch_output.non_tensor_batch.get("traj_index")).ravel().astype(np.int64)
        if success and "success_rate" in success:
            st = np.asarray(success["success_rate"])
            gen_batch_output.non_tensor_batch["success_per_traj"] = st[traj_idx]

        textcraft_records = (
            envs.get_textcraft_trajectory_records()
            if envs is not None and hasattr(envs, "get_textcraft_trajectory_records")
            else None
        )
        if textcraft_records is not None:
            n_rows = len(traj_idx)
            conversations = np.empty(n_rows, dtype=object)
            data_indices = np.empty(n_rows, dtype=object)
            item_ids = np.empty(n_rows, dtype=object)
            depths = np.empty(n_rows, dtype=object)
            for row_i, t_i in enumerate(traj_idx):
                rec = textcraft_records[int(t_i)]
                conversations[row_i] = rec.get("conversations", [])
                data_indices[row_i] = rec.get("data_idx", int(t_i))
                item_ids[row_i] = rec.get("item_id", f"textcraft_{data_indices[row_i]}")
                depths[row_i] = rec.get("depth")
            gen_batch_output.non_tensor_batch["textcraft_conversations"] = conversations
            gen_batch_output.non_tensor_batch["textcraft_data_idx"] = data_indices
            gen_batch_output.non_tensor_batch["textcraft_item_id"] = item_ids
            gen_batch_output.non_tensor_batch["textcraft_depth"] = depths

        # When dynamic management is on: add trajectory-derived keys **expanded to row-level**
        # so adjust_batch (select_idxs + concat) and balance_batch never see length mismatch.
        if enable_dynamic_management:
            num_rows = len(traj_idx)
            if wsm_arr is not None and wsm_arr.shape[0] == batch_size:
                gen_batch_output.non_tensor_batch["with_skills_mask"] = wsm_arr[traj_idx]
            if per_step_retrieved is not None:
                gen_batch_output.non_tensor_batch["per_step_retrieved_by_traj"] = np.array(
                    [per_step_retrieved[int(traj_idx[i])] for i in range(num_rows)], dtype=object
                )
        return gen_batch_output

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
        
        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """
        batch_size = len(gen_batch.batch)
        som_cfg = getattr(getattr(self.config, "env", None), "skills_only_memory", None) or {}
        # Collect per-step retrievals when using step-level skills (step_only or task_step); envs.retrieved_memories
        # is set in reset()/step() using that step's task+obs, so we snapshot after each step.
        mode = (som_cfg.get("skill_gen_mode") or "task_step").strip().lower()
        if mode not in ("task_only", "step_only", "task_step"):
            mode = "task_step"
        collect_per_step = (
            getattr(envs, "retrieval_memory", None) is not None
            and mode in ("step_only", "task_step")
        )
        per_step_retrieved: Optional[List[List[Dict]]] = [[] for _ in range(batch_size)] if collect_per_step else None

        # Initial observations from the environment
        obs, route_obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))
        original_obs = obs
        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"

        skill_text_mode = som_cfg.get("skill_text_for_retrieval", "full")
        if collect_per_step and envs.retrieved_memories is not None:
            for i in range(batch_size):
                per_step_retrieved[i].append({"step": 0, **_snapshot_retrieved_memories(envs.retrieved_memories[i], skill_text_mode)})
        
        # uid = one per "problem" group; traj_uid = one per trajectory. So group_size trajectories share one uid.
        rollout_n = int(getattr(self.config.env.rollout, 'n', 0) or 0)
        if rollout_n > 0:
            # Same uid for consecutive rollout_n indices (align with repeat interleave: traj 0..n-1 = group 0)
            num_groups = (batch_size + rollout_n - 1) // rollout_n
            group_uuids = [str(uuid.uuid4()) for _ in range(num_groups)]
            uid_batch = np.array([group_uuids[i // rollout_n] for i in range(batch_size)], dtype=object)
            if batch_size <= 64:  # only log when batch is small enough
                n_unique = len(set(uid_batch.tolist()))
                print(f"[Rollout] uid grouping: batch_size={batch_size}, rollout.n={rollout_n}, unique uids={n_unique} (expected {num_groups})")
        else:
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        is_done = np.zeros(batch_size, dtype=bool)
        fixed_eval_style_logging = _as_bool(
            self.config.get("trainer", {}).get("fixed_eval_style_logging", False)
        )
        fixed_eval_success = np.zeros(batch_size, dtype=bool)
        fixed_eval_log_path = None
        if fixed_eval_style_logging:
            fixed_eval_log_dir = self.config.get("trainer", {}).get("default_local_dir", "./outputs")
            os.makedirs(fixed_eval_log_dir, exist_ok=True)
            fixed_eval_log_path = os.path.join(fixed_eval_log_dir, "fixed_eval_validation.log")

            def _fixed_eval_log(line: str) -> None:
                print(line, flush=True)
                with open(fixed_eval_log_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        else:
            def _fixed_eval_log(line: str) -> None:
                return None
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        trainer_cfg = self.config.get("trainer", {})
        dump_random_trace_cfg = trainer_cfg.get("dump_random_trace_json", False)
        dump_random_trace_mode = (
            dump_random_trace_cfg.strip().lower()
            if isinstance(dump_random_trace_cfg, str)
            else ""
        )
        dump_random_trace = _as_bool(dump_random_trace_cfg) or dump_random_trace_mode in {
            "once",
            "train_once",
        }
        if dump_random_trace_mode == "train_once" and not is_train:
            dump_random_trace = False
        if dump_random_trace_mode in {"once", "train_once"} and self._random_trace_dumped:
            dump_random_trace = False
        if dump_random_trace and batch_size > 0:
            trace_start = batch_size // 3
            trace_end = max(trace_start + 1, (batch_size * 2) // 3)
            trace_index = int(np.random.randint(trace_start, trace_end))
        else:
            trace_index = None
        trace_steps = []
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        api_costs = np.zeros(batch_size, dtype=np.float32)
        # route_obs = obs
        # original_obs = obs
        # Trajectory collection loop
        for _step in range(self.config.env.max_steps):
            if fixed_eval_style_logging:
                _fixed_eval_log(
                    f"[FixedEval] Step {_step}; Dones ({int(is_done.sum())}/{batch_size}); "
                    f"SR {float(fixed_eval_success.mean()):.4f}"
                )
            active_masks = np.logical_not(is_done)

            batch = self.preprocess_batch(gen_batch=gen_batch, obs=route_obs)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info

            forced_route_model = self._get_forced_route_model()
            force_model_enabled = bool(forced_route_model)
            if force_model_enabled:
                batch_output = self._build_forced_router_batch_output(batch_input)
            else:
                # pad to be divisible by dp_size
                batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
                batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
                # # unpad
                
                batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid

            batch = batch.union(batch_output)
            if force_model_enabled:
                route_actions_str = [
                    f"<think>Forced fixed route evaluation.</think><search>{forced_route_model}</search>"
                ] * len(batch.batch["responses"])
            else:
                route_actions_str = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
            # print(f'路由器输出：{route_actions_str}')
            cur_completion_tokens, text_model_actions, models, route_format_valid, model_call_success, model_call_elapsed_seconds = self.execute_predictions(
                route_actions_str, original_obs, active_mask=active_masks
            )
            route_format_valid = np.array(route_format_valid, dtype=bool)
            model_call_success = np.array(model_call_success, dtype=bool)
            route_actions_for_record = (
                [
                    f"<think>Forced fixed route evaluation.</think><search>{forced_route_model}</search>"
                ] * len(route_actions_str)
                if forced_route_model
                else [self._router_action_for_record(action) for action in route_actions_str]
            )
            if dump_random_trace and trace_index is not None and trace_index < batch_size and active_masks[trace_index]:
                routed_user_prompt = original_obs["text"][trace_index] if original_obs.get("text") is not None else None
                routed_chat_prompts = original_obs.get("chat", None)
                routed_chat_prompt = (
                    routed_chat_prompts[trace_index]
                    if routed_chat_prompts is not None and trace_index < len(routed_chat_prompts)
                    else None
                )
                routed_system_prompts = original_obs.get("system", None)
                routed_system_prompt = (
                    routed_system_prompts[trace_index]
                    if routed_system_prompts is not None and trace_index < len(routed_system_prompts)
                    else ""
                )
                routed_model_prompt_for_trace = (
                    routed_chat_prompt
                    if routed_chat_prompt is not None
                    else (
                        f"System prompt:\n{routed_system_prompt}\n\nUser prompt:\n{routed_user_prompt}"
                        if routed_system_prompt
                        else routed_user_prompt
                    )
                )
                trace_steps.append({
                    "router_prompt": route_obs["text"][trace_index] if route_obs.get("text") is not None else None,
                    "router_output": route_actions_for_record[trace_index] if trace_index < len(route_actions_for_record) else "",
                    "routed_model_prompt": routed_model_prompt_for_trace,
                    "routed_model_output": text_model_actions[trace_index] if trace_index < len(text_model_actions) else "",
                })
            batch.non_tensor_batch['router_actions'] = np.array(
                route_actions_for_record,
                dtype=object,
            )
            batch.non_tensor_batch['route_format_valid'] = route_format_valid
            batch.non_tensor_batch['model_actions'] = np.array(text_model_actions, dtype=object)
            batch.non_tensor_batch['called_models'] = np.array(models, dtype=object)
            batch.non_tensor_batch['model_call_success'] = model_call_success
            batch.non_tensor_batch['model_call_elapsed_seconds'] = np.asarray(model_call_elapsed_seconds, dtype=np.float32)
            batch.non_tensor_batch['step_input_tokens'] = np.asarray(step_prompt_tokens, dtype=np.float32)
            batch.non_tensor_batch['step_output_tokens'] = np.asarray(step_completion_tokens, dtype=np.float32)
            batch.non_tensor_batch['step_total_tokens'] = np.asarray(step_total_tokens, dtype=np.float32)
            batch = self.keep_router_outputs_only(batch, route_actions_for_record, text_model_actions)
            # print(f"路由模型执行动作：{text_model_actions}")
            next_obs, next_route_obs, rewards, dones, infos = envs.step(text_model_actions, models)

            if collect_per_step and envs.retrieved_memories is not None:
                for i in range(batch_size):
                    per_step_retrieved[i].append({
                        "step": _step + 1,
                        **_snapshot_retrieved_memories(envs.retrieved_memories[i], skill_text_mode),
                    })
            
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['env_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['env_action_valid'] = np.ones(batch_size, dtype=bool)
            batch.non_tensor_batch['is_action_valid'] = route_format_valid
            batch.non_tensor_batch['alfworld_task'] = np.array(
                [_alfworld_task_from_info(info) for info in infos],
                dtype=object,
            )

            if 'tool_calling' in infos[0]:
                tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]
            # Create reward tensor, only assign rewards for active environments
            # episode_rewards += torch_to_numpy(rewards) * torch_to_numpy(active_masks)
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1
            api_costs[active_masks] += np.asarray(cur_completion_tokens, dtype=np.float32)[active_masks]

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['step_api_costs'] = np.asarray(cur_completion_tokens, dtype=np.float32).copy()
            batch.non_tensor_batch['api_costs'] = api_costs.copy()
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
            # Update episode lengths for active environments
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            # Update done states
            newly_done = np.logical_and(np.logical_not(is_done), dones)
            if fixed_eval_style_logging and np.any(newly_done):
                for i in np.where(newly_done)[0]:
                    fixed_eval_success[i] = bool(infos[i].get("won", False))
            is_done = np.logical_or(is_done, dones)
                
            # Update observations for next step
            route_obs = next_route_obs
            original_obs = next_obs
            # Break if all environments are done
            if is_done.all():
                if fixed_eval_style_logging:
                    _fixed_eval_log(
                        f"[FixedEval] Finished early at step {_step}; "
                        f"SR {float(fixed_eval_success.mean()):.4f}"
                    )
                break
        
        if dump_random_trace and trace_index is not None:
            save_dir = self.config.get("trainer", {}).get("default_local_dir", "./outputs")
            os.makedirs(save_dir, exist_ok=True)
            trace_payload = {
                "steps": trace_steps,
            }
            trace_path = os.path.join(save_dir, "random_trace.json")
            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump(trace_payload, f, indent=2, ensure_ascii=False)
            self._random_trace_dumped = True
            print(f"[TraceDump] Wrote random trajectory trace to {trace_path}", flush=True)

        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )
        wsm_traj = getattr(envs, "with_skills_mask", None)
        if wsm_traj is not None:
            wsm_traj = np.asarray(wsm_traj, dtype=bool).copy()

        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings, api_costs, per_step_retrieved, wsm_traj
    
    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        total_tool_callings = []
        total_api_costs = []
        total_wsm_chunks: List[np.ndarray] = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings, api_costs, _, wsm_traj = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                is_train=True,
            )
            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings, wsm_traj, api_costs = filter_group_data(batch_list=batch_list,
                                                                                                episode_rewards=episode_rewards, 
                                                                                                episode_lengths=episode_lengths, 
                                                                                                success=success, 
                                                                                                traj_uid=traj_uid, 
                                                                                                tool_callings=tool_callings, 
                                                                                                config=self.config,
                                                                                                last_try=(try_count == max_try_count),
                                                                                                with_skills_per_traj=wsm_traj,
                                                                                                api_costs=api_costs,
                                                                                                )
            
            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_tool_callings.append(tool_callings)
            total_api_costs.append(api_costs)
            if wsm_traj is not None:
                total_wsm_chunks.append(np.asarray(wsm_traj, dtype=bool))

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)
        total_api_costs = np.concatenate(total_api_costs, axis=0)
        total_wsm = np.concatenate(total_wsm_chunks, axis=0) if total_wsm_chunks else None

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings, total_api_costs, total_wsm

    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        if is_train:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
            
        # Initial observations from the environment
        per_step_retrieved = None
        total_wsm: Optional[np.ndarray] = None
        if self.config.algorithm.filter_groups.enable and is_train:
            # Dynamic Sampling (for DAPO and Dynamic GiGPO)
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings, total_api_costs, total_wsm = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            per_step_retrieved = None
        else:
            # Vanilla Sampling   
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings, total_api_costs, per_step_retrieved, total_wsm = \
                self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                is_train=is_train,
            )
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        assert len(total_batch_list) == len(totoal_tool_callings)
        assert len(total_batch_list) == len(total_api_costs)
        

        # Create trajectory data
        som_cfg = (self.config.env.get("skills_only_memory") or {}) if hasattr(self.config, "env") else {}
        enable_dynamic_management = bool(som_cfg.get("enable_dynamic_management", False))
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=totoal_tool_callings,
            api_costs=total_api_costs,
            per_step_retrieved=per_step_retrieved,
            envs=envs,
            enable_dynamic_management=enable_dynamic_management,
            with_skills_per_traj=total_wsm,
        )
        
        return gen_batch_output
    
    def execute_predictions(self, predictions: List[str], original_obs: Dict, do_route=True, active_mask=None) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            predictions: List of router predictions
            original_obs: Current environment observations used as routed-model query context
        """
        model_actions = []
        models = []
        model_call_success = []
        model_call_elapsed_seconds = []
        # 预处理router输出结果，获得content = model:query 
        #但现在只有model
        contexts = original_obs.get('text', None)
        forced_route_model = self._get_forced_route_model()
        if active_mask is None:
            active_mask = [True] * len(predictions)
        else:
            active_mask = [bool(x) for x in active_mask]
        if forced_route_model:
            cur_actions = ['search'] * len(predictions)
            contents = [forced_route_model] * len(predictions)
            route_format_valids = [True] * len(predictions)
        else:
            cur_actions, contents, route_format_valids = self.postprocess_predictions(predictions)
        for idx, is_active in enumerate(active_mask):
            if not is_active:
                cur_actions[idx] = ''
                contents[idx] = ''
                route_format_valids[idx] = True
        # print(f"contents: {contents}")
        cur_completion_tokens = []
        step_prompt_tokens = []
        step_completion_tokens = []
        step_total_tokens = []
        # 构造agent的content
        # route_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        # route_queries = [
        #     f"{content}:{context}"
        #     for action, context, content in zip(cur_actions, contexts, contents)
        #     if action == 'search'
        # ]
        # print(f"route_queries:{route_queries}")
        # route_queries = {
        #     "model_name": contents,
        #     "query": contexts
        # }
        system_contexts = original_obs.get('system', None)
        chat_contexts = original_obs.get('chat', None)
        route_query_payloads = []
        route_system_payloads = []
        for idx, action in enumerate(cur_actions):
            if action != 'search':
                continue
            if chat_contexts is not None and idx < len(chat_contexts):
                route_query_payloads.append(chat_contexts[idx])
                route_system_payloads.append("")
            else:
                route_query_payloads.append(contexts[idx])
                route_system_payloads.append(
                    system_contexts[idx]
                    if system_contexts is not None and idx < len(system_contexts)
                    else ""
                )
        route_queries = {
            "model_name": [
                forced_route_model or content
                for action, content in zip(cur_actions, contents)
                if action == 'search'
            ],
            "query": route_query_payloads,
            "system": route_system_payloads,
        }
        if do_route:
            route_results, completion_tokens_list, called_model_names, called_model_elapsed_seconds, token_usage_list = self.batch_route(route_queries)
            assert len(route_results) == sum([1 for action in cur_actions if action == 'search'])
            assert len(route_results) == len(completion_tokens_list)
            assert len(route_results) == len(called_model_names)
            assert len(route_results) == len(called_model_elapsed_seconds)
            assert len(route_results) == len(token_usage_list)
        else:
            route_results = [''] * sum([1 for action in cur_actions if action == 'search'])
            completion_tokens_list = [0.0] * sum([1 for action in cur_actions if action == 'search'])
            called_model_names = [''] * sum([1 for action in cur_actions if action == 'search'])
            called_model_elapsed_seconds = [0.0] * sum([1 for action in cur_actions if action == 'search'])
            token_usage_list = [{"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}] * sum([1 for action in cur_actions if action == 'search'])

        for action in cur_actions:
            if action == 'search':
                called_model_name = called_model_names.pop(0)
                elapsed_seconds = called_model_elapsed_seconds.pop(0)
                token_usage = token_usage_list.pop(0)
                route_result = route_results.pop(0)
                route_result_lower = route_result.strip().lower()
                if route_result_lower == "llm name error":
                    model_actions.append('')
                    models.append('')
                    model_call_success.append(False)
                elif route_result_lower == "api request error":
                    model_actions.append('')
                    models.append(called_model_name)
                    model_call_success.append(False)
                else:
                    model_actions.append(route_result.strip())
                    models.append(called_model_name)
                    model_call_success.append(True)
                cur_completion_tokens.append(completion_tokens_list.pop(0))
                step_prompt_tokens.append(float((token_usage or {}).get("prompt_tokens", 0) or 0))
                step_completion_tokens.append(float((token_usage or {}).get("completion_tokens", 0) or 0))
                step_total_tokens.append(float((token_usage or {}).get("total_tokens", 0) or 0))
                model_call_elapsed_seconds.append(float(elapsed_seconds or 0.0))
            else:
                model_actions.append('')
                models.append('')
                model_call_success.append(False)
                cur_completion_tokens.append(0.0)
                step_prompt_tokens.append(0.0)
                step_completion_tokens.append(0.0)
                step_total_tokens.append(0.0)
                model_call_elapsed_seconds.append(0.0)
        # print(f'len(route_results): {len(route_results)}')
        # print(f'len(completion_tokens_list): {len(completion_tokens_list)}')
        # print(f'len(called_model_names): {len(called_model_names)}')
        assert len(route_results) == 0
        assert len(completion_tokens_list) == 0
        assert len(called_model_names) == 0
        assert len(called_model_elapsed_seconds) == 0
        assert len(token_usage_list) == 0
        return cur_completion_tokens, model_actions, models, route_format_valids, model_call_success, model_call_elapsed_seconds, step_prompt_tokens, step_completion_tokens, step_total_tokens

    def _get_forced_route_model(self) -> str:
        """Optional eval hook: force every routed search call to a fixed backend model."""
        routing_cfg = self.config.get("routing", {}) if hasattr(self.config, "get") else {}
        enabled = bool(routing_cfg.get("force_model_enable", False))
        if not enabled:
            return ""
        model_name = str(routing_cfg.get("force_model_name", "")).strip()
        if not model_name:
            raise ValueError("routing.force_model_enable=True requires routing.force_model_name to be set")
        llm_name, _ = check_llm_name(model_name)
        if not llm_name:
            raise ValueError(f"Unknown routing.force_model_name: {model_name}")
        return model_name

    @staticmethod
    def _postprocess_route_action(action: str) -> str:
        if "</search>" in action:
            return action.split("</search>", 1)[0] + "</search>"
        if "</answer>" in action:
            return action.split("</answer>", 1)[0] + "</answer>"
        return action

    @staticmethod
    def _router_action_for_record(action: str) -> str:
        return TrajectoryCollector._postprocess_route_action(action or "").strip()

    @staticmethod
    def _project_route_action(prediction: str) -> Tuple[str, str, bool]:
        original = (prediction or "").strip()
        trimmed = TrajectoryCollector._postprocess_route_action(original)
        pattern = r"\A<think>\s*(.*?)\s*</think>\s*<search>\s*(.*?)\s*</search>\Z"
        match = re.fullmatch(pattern, trimmed, re.DOTALL)
        if not match:
            return None, "", False
        reasoning = match.group(1).strip()
        model_name = match.group(2).strip()
        llm_name, _ = check_llm_name(model_name)
        valid = (
            original == trimmed
            and bool(reasoning)
            and llm_name in MODEL_CONF
            and len(re.findall(r"<think>", original)) == 1
            and len(re.findall(r"</think>", original)) == 1
            and len(re.findall(r"<search>", original)) == 1
            and len(re.findall(r"</search>", original)) == 1
            and not re.search(r"</?answer>", original)
        )
        return ("search" if valid else None), (llm_name if valid else ""), valid

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[str], List[str], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
        valids = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                action, content, valid = self._project_route_action(prediction)
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            valids.append(valid)
            
        return actions, contents, valids
    def batch_route(self, queries: Dict = None) -> str:
        ret = access_routing_pool(queries=queries, api_base=self.config.api_base, api_key=self.config.api_key)
        
        return ret['result'], ret["completion_tokens_list"], ret.get("called_model_names", []), ret.get("model_call_elapsed_seconds", []), ret.get("token_usage_list", [])
