# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import os
import re
import warnings
from collections import defaultdict
from contextlib import contextmanager
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.worker.gpu_worker import Worker as BaseWorker

import modelopt.torch.quantization as mtq
from modelopt.torch.utils.dataset_utils import get_dataset_dataloader


def convert_amax_hf2vllm(
    hf_state_dict: dict[str, torch.Tensor], fuse_experts: bool = False
) -> tuple[dict[str, torch.Tensor], dict[str, list[int]]]:
    """
    Convert amax values from HuggingFace format to vLLM format.

    This function merges:
    - q_proj, k_proj, v_proj amax values into qkv_proj (taking max)
    - gate_proj, up_proj amax values into gate_up_proj (taking max)

    Args:
        hf_state_dict: HuggingFace state dict containing amax values

    Returns:
        Tuple of (vLLM format state dict with merged amax values,
                  dict mapping merged keys to component sizes for TP sharding)
    """
    vllm_state_dict = {}
    # Track component sizes for merged projections (needed for proper TP sharding)
    # e.g., {"model.layers.0.self_attn.qkv_proj.weight_quantizer._amax": [q_size, k_size, v_size]}
    merged_component_sizes: dict[str, list[int]] = {}

    # Group keys by their base pattern (without the specific projection name)
    merge_groups = defaultdict(list)

    for key, value in hf_state_dict.items():
        if "_amax" not in key:
            # Copy non-amax keys as-is
            vllm_state_dict[key] = value
            continue

        # Check if this is a q/k/v projection that needs merging
        qkv_match = re.search(r"(.*\.)([qkv])_proj(\..+_amax)$", key)
        if qkv_match:
            base_pattern = qkv_match.group(1) + "qkv_proj" + qkv_match.group(3)
            merge_groups[base_pattern].append((key, value))
            continue

        # Check if this is an expert gate/up projection
        # Pattern: model.layers.0.mlp.experts.*.gate_proj.input_quantizer._amax and
        # model.layers.0.mlp.experts.*.up_proj.input_quantizer._amax
        # Maps to: model.layers.0.mlp.experts.w13_input_quantizer._amax
        expert_gate_up_match = (
            "mixer" not in key
            and fuse_experts
            and re.search(r"(.*\.experts)\.\d+\.(gate|up)_proj\.([^.]+_quantizer\._amax)$", key)
        )
        if expert_gate_up_match:
            base_pattern = expert_gate_up_match.group(1) + ".w13_" + expert_gate_up_match.group(3)
            merge_groups[base_pattern].append((key, value))
            continue

        # Check if this is a non-expert gate/up projection that needs merging
        gate_up_match = (
            "mixer" not in key
            and "experts" not in key
            and re.search(r"(.*\.)(gate|up)_proj(\..+_amax)$", key)
        )
        if gate_up_match:
            base_pattern = gate_up_match.group(1) + "gate_up_proj" + gate_up_match.group(3)
            merge_groups[base_pattern].append((key, value))
            continue

        # Check if this is an expert down_proj
        # Pattern: model.layers.0.mlp.experts.*.down_proj.input_quantizer._amax
        # Maps to: model.layers.0.mlp.experts.w2_input_quantizer._amax
        expert_down_match = (
            "mixer" not in key
            and fuse_experts
            and re.search(r"(.*\.experts)\.\d+\.down_proj\.([^.]+_quantizer\._amax)$", key)
        )
        if expert_down_match:
            base_pattern = expert_down_match.group(1) + ".w2_" + expert_down_match.group(2)
            merge_groups[base_pattern].append((key, value))
            continue

        # Copy other amax keys as-is (like o_proj, down_proj)
        vllm_state_dict[key] = value

    # Define the expected order for merged projections
    # vLLM expects: qkv_proj = [Q, K, V], gate_up_proj = [gate, up]
    def get_sort_key(key_value_tuple):
        key = key_value_tuple[0]
        # For QKV projections: ensure order is q, k, v
        if "q_proj" in key:
            return 0
        elif "k_proj" in key:
            return 1
        elif "v_proj" in key:
            return 2
        # For gate/up projections: ensure order is gate, up
        elif "gate_proj" in key or "gate" in key:
            return 0
        elif "up_proj" in key or "up" in key:
            return 1
        # Default: sort alphabetically
        return key

    # Merge grouped amax values
    for merged_key, key_value_pairs in merge_groups.items():
        if len(key_value_pairs) > 1:
            # Sort to ensure correct concatenation order
            key_value_pairs_sorted = sorted(key_value_pairs, key=get_sort_key)
            values = [value for _, value in key_value_pairs_sorted]
            shapes = [v.shape for v in values]
            is_weight_quantizer = "weight_quantizer" in merged_key

            # Check if all values are scalars (0-dimensional tensors)
            # This happens with per-tensor quantization (e.g., NVFP4_DEFAULT_CFG)
            all_scalars = all(v.ndim == 0 for v in values)

            if is_weight_quantizer:
                # Weight quantizers: vLLM fuses weights by concatenation
                # (qkv_proj = concat(q, k, v), gate_up_proj = concat(gate, up))
                if all_scalars:
                    # For scalar amax (per-tensor quantization), take max after fusing
                    # because the fused weight needs a single amax that covers all parts
                    merged_value = torch.stack(values).max()
                    vllm_state_dict[merged_key] = merged_value
                    print(
                        f"Merged {len(key_value_pairs)} scalar weight amax keys into {merged_key} "
                        f"(taking max of scalars -> scalar)"
                    )
                else:
                    # For per-channel amax, concatenate along the channel dimension
                    merged_value = torch.cat(values, dim=0)
                    vllm_state_dict[merged_key] = merged_value
                    # Store component sizes for proper TP sharding later
                    # vLLM shards each component (Q/K/V or gate/up) separately
                    merged_component_sizes[merged_key] = [v.shape[0] for v in values]
                    print(
                        f"Concatenated {len(key_value_pairs_sorted)} weight amax keys into {merged_key} "
                        f"(shapes {shapes} -> {merged_value.shape})"
                    )
                for orig_key, _ in key_value_pairs_sorted:
                    print(f"  - {orig_key}")
            # Input quantizers: take max (they share the same input tensor)
            elif all_scalars or all(s == shapes[0] for s in shapes):
                merged_value = torch.stack(values).max(dim=0)[0]
                vllm_state_dict[merged_key] = merged_value
                print(f"Merged {len(key_value_pairs_sorted)} input amax keys into {merged_key}")
                for orig_key, _ in key_value_pairs_sorted:
                    print(f"  - {orig_key}")
            else:
                # Different shapes for input quantizers - this shouldn't happen normally
                # but handle it gracefully by taking element-wise max after padding
                merged_value = torch.stack(values).max(dim=0)[0]
                vllm_state_dict[merged_key] = merged_value
                print(
                    f"Warning: Input quantizer amax shapes differ {shapes}, "
                    f"taking max for {merged_key}"
                )
                for orig_key, _ in key_value_pairs_sorted:
                    print(f"  - {orig_key}")
        else:
            # Single key, just rename it
            _, value = key_value_pairs[0]
            vllm_state_dict[merged_key] = value

    return vllm_state_dict, merged_component_sizes


@contextmanager
def disable_compilation(model):
    do_not_compile = True
    if hasattr(model, "model"):
        do_not_compile = model.model.do_not_compile
        model.model.do_not_compile = True
    elif hasattr(model, "language_model"):
        do_not_compile = model.language_model.model.do_not_compile
        model.language_model.model.do_not_compile = True
    else:
        raise ValueError("Model does not have a model or language_model attribute")

    try:
        yield
    finally:
        if hasattr(model, "model"):
            model.model.do_not_compile = do_not_compile
        elif hasattr(model, "language_model"):
            model.language_model.model.do_not_compile = do_not_compile


quant_config: dict[str, Any] = {
    "dataset": os.environ.get("QUANT_DATASET", "cnn_dailymail"),
    "calib_size": int(os.environ.get("QUANT_CALIB_SIZE", 512)),
    "quant_cfg": os.environ.get("QUANT_CFG", None),
    "kv_quant_cfg": os.environ.get("KV_QUANT_CFG", None),
    "amax_file_path": os.environ.get("AMAX_FILE_PATH", None),
}


def update_kv_cfg_for_mla(model: torch.nn.Module, kv_quant_cfg: dict[str, Any]) -> dict[str, Any]:
    """Update KV cache quantization config for MLA models.

    MLA uses `kv_c_bmm_quantizer` (compressed KV) instead of separate
    `k_bmm_quantizer` and `v_bmm_quantizer`. This function copies the
    config from `*[kv]_bmm_quantizer` to also cover `*kv_c_bmm_quantizer`.
    """
    try:
        from vllm.attention.layer import MLAAttention
    except ImportError:
        return kv_quant_cfg

    if not any(isinstance(m, MLAAttention) for m in model.modules()):
        return kv_quant_cfg

    if kv_config := kv_quant_cfg.get("*[kv]_bmm_quantizer"):
        kv_quant_cfg["*kv_c_bmm_quantizer"] = kv_config
        kv_quant_cfg["*k_pe_bmm_quantizer"] = kv_config
        print("MLA detected: added *kv_c_bmm_quantizer and k_pe_bmm_quantizer config")

    return kv_quant_cfg


def _create_new_data_cls(data_cls, **kwargs):
    """vLLM's low-level API changes frequently. This function creates a class with parameters
    compatible with the different vLLM versions."""
    valid_params = {field.name for field in dataclasses.fields(data_cls)}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return data_cls(**filtered_kwargs)


def _fakequant_run_prolog_worker(self) -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        self.model_runner.model_config.tokenizer,
        trust_remote_code=True,
    )
    if tokenizer.pad_token != "<unk>" or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quant_config["amax_file_path"]:
        print("Will load amax, so only do a single sample calibration")
        quant_config["calib_size"] = 1

    calib_dataloader = get_dataset_dataloader(
        dataset_name=quant_config["dataset"],
        tokenizer=tokenizer,
        batch_size=1,
        num_samples=quant_config["calib_size"],
        device=self.device,
    )

    def calibrate_loop(model: Any = None) -> None:
        for batch_idx, batch in tqdm(enumerate(calib_dataloader)):
            input_ids = batch["input_ids"][0]

            # Convert tensor to list of integers for vLLM compatibility
            if torch.is_tensor(input_ids):
                input_ids_list = input_ids.cpu().tolist()
            else:
                input_ids_list = list(input_ids)

            num_groups = len(self.model_runner.kv_cache_config.kv_cache_groups)
            empty_block_ids = tuple([] for _ in range(num_groups))

            req_id = f"req-{batch_idx}"
            # Pass all possible parameters - the helper will filter based on vLLM version
            new_req = _create_new_data_cls(
                NewRequestData,
                req_id=req_id,
                prompt_token_ids=input_ids_list,
                # Old API parameters
                mm_kwargs=[],  # TODO: remove this when vllm <= 0.11 is outdated
                mm_hashes=[],  # TODO: remove this when vllm <= 0.11 is outdated
                mm_positions=[],  # TODO: remove this when vllm <= 0.11 is outdated
                # New API parameter
                mm_features=[],
                sampling_params=SamplingParams(max_tokens=1),
                pooling_params=None,
                block_ids=empty_block_ids,
                num_computed_tokens=0,
                lora_request=None,
            )

            scheduler_output = _create_new_data_cls(
                SchedulerOutput,
                scheduled_new_reqs=[new_req],
                scheduled_cached_reqs=CachedRequestData.make_empty(),
                num_scheduled_tokens={req_id: len(input_ids_list)},
                total_num_scheduled_tokens=len(input_ids_list),
                scheduled_spec_decode_tokens={},
                scheduled_encoder_inputs={},
                num_common_prefix_blocks=[0] * num_groups,
                finished_req_ids=set(),
                free_encoder_mm_hashes=[],
                kv_connector_metadata=None,
                # Old API parameters
                structured_output_request_ids={},  # TODO: remove this when vllm <= 0.11 is outdated
                grammar_bitmask=None,  # TODO: remove this when vllm <= 0.11 is outdated
            )
            output = self.execute_model(scheduler_output)
            if hasattr(self, "sample_tokens"):
                if output is None:  # TODO: make this default when vllm <= 0.11 is outdated
                    self.sample_tokens(None)

    quant_cfg = {} if quant_config["quant_cfg"] is None else getattr(mtq, quant_config["quant_cfg"])
    quant_kv_cfg = (
        {} if quant_config["kv_quant_cfg"] is None else getattr(mtq, quant_config["kv_quant_cfg"])
    )

    # When loading from amax file, override algorithm to "max" since calibration was done offline.
    amax_file_path = quant_config["amax_file_path"]
    if amax_file_path and quant_cfg:
        original_algorithm = quant_cfg.get("algorithm")
        if isinstance(original_algorithm, dict) or original_algorithm not in ["max", None]:
            print(
                f"Overriding algorithm from {original_algorithm} to 'max' since loading from amax file"
            )
            quant_cfg = {**quant_cfg, "algorithm": "max"}

    model = self.model_runner.model
    if hasattr(model, "unwrap"):
        model = model.unwrap()

    # Check if model has MLA and update KV config accordingly
    if quant_kv_cfg:
        quant_kv_cfg["quant_cfg"] = update_kv_cfg_for_mla(model, quant_kv_cfg["quant_cfg"])

    if quant_kv_cfg:
        quant_cfg = mtq.utils.update_quant_cfg_with_kv_cache_quant(
            quant_cfg, quant_kv_cfg["quant_cfg"]
        )

    with disable_compilation(model):
        print("quantizing model...")
        mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)

    amax_file_path = quant_config["amax_file_path"]
    if amax_file_path:
        print(f"Loading amax values from {amax_file_path}")
        saved_amax_dict = torch.load(amax_file_path)
        # convert amax keys to vLLM format
        if hasattr(self.model_runner.model, "hf_to_vllm_mapper"):
            saved_amax_dict = self.model_runner.model.hf_to_vllm_mapper.apply_dict(saved_amax_dict)
            saved_amax_dict = {
                key.replace("quantizer_amax", "quantizer._amax"): value
                for key, value in saved_amax_dict.items()
                if key.endswith("quantizer_amax")
            }
        saved_amax_dict, merged_component_sizes = convert_amax_hf2vllm(saved_amax_dict, fuse_experts=True)

        current_state_dict = model.state_dict()
        # Count amax keys in checkpoint and model
        checkpoint_amax_keys = [key for key in saved_amax_dict if key.endswith("_amax")]
        model_amax_keys = [key for key in current_state_dict if key.endswith("_amax")]
        for key in checkpoint_amax_keys:
            if key not in model_amax_keys:
                print(f"Key {key} not found in model state dict, but exists in checkpoint")
        for key in model_amax_keys:
            if key not in checkpoint_amax_keys:
                raise ValueError(
                    f"Key {key} not found in checkpoint state dict, but exists in model"
                )

        checkpoint_amax_count = len(checkpoint_amax_keys)
        model_amax_count = len(model_amax_keys)

        # Ensure counts match
        if checkpoint_amax_count != model_amax_count:
            warnings.warn(
                f"Mismatch in amax key counts: checkpoint has {checkpoint_amax_count} "
                f"amax keys but model has {model_amax_count} amax keys. This can happen if the model is using PP."
            )

        # Update amax values with tensor parallelism support
        # When using TP, amax values need to be sharded to match the model's weight sharding
        tp_size = 1
        tp_rank = 0
        if torch.distributed.is_initialized():
            from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
            tp_size = get_tensor_model_parallel_world_size()
            tp_rank = get_tensor_model_parallel_rank()

        for key, value in saved_amax_dict.items():
            if key in current_state_dict:
                model_shape = current_state_dict[key].shape
                ckpt_shape = value.shape

                # Handle tensor parallelism sharding for non-scalar amax
                if model_shape != ckpt_shape and value.ndim > 0:
                    # The checkpoint has full (unsharded) amax, but model has TP-sharded amax
                    # Shard the checkpoint amax to match the model's sharding
                    if tp_size > 1 and ckpt_shape[0] == model_shape[0] * tp_size:
                        # Check if this is a merged projection (qkv_proj, gate_up_proj)
                        # that needs component-wise sharding
                        if key in merged_component_sizes:
                            # For merged projections, vLLM shards EACH component separately
                            # e.g., qkv_proj amax = [Q_amax, K_amax, V_amax]
                            # Rank 0 should get: [Q_amax_first_half, K_amax_first_half, V_amax_first_half]
                            # NOT: [Q_amax_all, K_amax_none, V_amax_none]
                            component_sizes = merged_component_sizes[key]

                            # Split the concatenated amax into components
                            components = torch.split(value, component_sizes, dim=0)

                            # Shard each component separately
                            sharded_components = []
                            for comp in components:
                                comp_shard_size = comp.shape[0] // tp_size
                                start_idx = tp_rank * comp_shard_size
                                end_idx = start_idx + comp_shard_size
                                sharded_components.append(comp[start_idx:end_idx])

                            # Re-concatenate the sharded components
                            value = torch.cat(sharded_components, dim=0)
                            print(
                                f"Sharded merged amax {key} for TP rank {tp_rank}: "
                                f"{ckpt_shape} -> {value.shape} "
                                f"(component sizes: {component_sizes} -> {[c.shape[0] for c in sharded_components]})"
                            )
                        else:
                            # Check if this is a row-parallel layer (o_proj, down_proj)
                            # Row-parallel layers need strided/interleaved sharding
                            is_row_parallel = any(x in key for x in ["o_proj", "down_proj"])

                            if is_row_parallel:
                                # For row-parallel layers (o_proj, down_proj):
                                # - Weight is sharded along INPUT dimension (columns)
                                # - Amax is organized as [out_feat_0_blocks, out_feat_1_blocks, ...]
                                # - Each rank needs HALF the blocks for EACH output feature
                                #
                                # Example for o_proj with shape [2048, 2048], block_size=16, TP=2:
                                # - Full amax: [262144, 1] = [2048 out_feat * 128 blocks_per_out, 1]
                                # - Rank 0 needs: blocks 0-63 for each output feature
                                # - Rank 1 needs: blocks 64-127 for each output feature
                                # - Sharded amax: [131072, 1] = [2048 * 64, 1]

                                # Compute the number of blocks per output feature
                                full_size = ckpt_shape[0]
                                sharded_size = model_shape[0]
                                # full_size = num_out_feat * blocks_per_out
                                # sharded_size = num_out_feat * blocks_per_out / tp_size
                                # So: num_out_feat = sharded_size * tp_size / (tp_size - 1 + tp_size)
                                # Actually: full_size / sharded_size = tp_size
                                # And: full_size = num_out * full_blocks_per_out
                                #      sharded_size = num_out * sharded_blocks_per_out
                                # So: full_blocks_per_out / sharded_blocks_per_out = tp_size
                                #     sharded_blocks_per_out = full_blocks_per_out / tp_size

                                # We need to find num_out_feat and blocks_per_out
                                # Since full_size = num_out * blocks_per_out, we need one more constraint
                                # The constraint is: blocks_per_out must be divisible by tp_size
                                # Common values: blocks_per_out = 128 (hidden_dim=2048, block_size=16)

                                # Find the best factorization
                                # We know sharded_blocks_per_out = blocks_per_out / tp_size
                                # Try common block counts: 128, 64, 256, 384, 512
                                found_factorization = False
                                for full_blocks_per_out in [128, 256, 384, 512, 64, 192, 320]:
                                    if full_size % full_blocks_per_out == 0:
                                        num_out_feat = full_size // full_blocks_per_out
                                        if full_blocks_per_out % tp_size == 0:
                                            sharded_blocks_per_out = full_blocks_per_out // tp_size
                                            expected_sharded = num_out_feat * sharded_blocks_per_out
                                            if expected_sharded == sharded_size:
                                                found_factorization = True
                                                break

                                if not found_factorization:
                                    print(
                                        f"Warning: Could not find valid factorization for {key}. "
                                        f"Falling back to contiguous sharding."
                                    )
                                    shard_size = model_shape[0]
                                    start_idx = tp_rank * shard_size
                                    end_idx = start_idx + shard_size
                                    value = value[start_idx:end_idx]
                                else:
                                    # Reshape to [num_out_feat, blocks_per_out]
                                    value_2d = value.view(num_out_feat, full_blocks_per_out)

                                    # Select the blocks for this rank
                                    start_block = tp_rank * sharded_blocks_per_out
                                    end_block = start_block + sharded_blocks_per_out
                                    value_sharded = value_2d[:, start_block:end_block]

                                    # Flatten back to [num_out_feat * sharded_blocks_per_out, 1]
                                    value = value_sharded.contiguous().view(-1, 1)

                                    print(
                                        f"Sharded row-parallel amax {key} for TP rank {tp_rank}: "
                                        f"{ckpt_shape} -> {value.shape} "
                                        f"(num_out={num_out_feat}, "
                                        f"blocks {start_block}:{end_block} of {full_blocks_per_out})"
                                    )
                            else:
                                # Column-parallel layers: simple contiguous sharding
                                shard_size = model_shape[0]
                                start_idx = tp_rank * shard_size
                                end_idx = start_idx + shard_size
                                value = value[start_idx:end_idx]
                                print(
                                    f"Sharded column-parallel amax {key} for TP rank {tp_rank}: "
                                    f"{ckpt_shape} -> {value.shape}"
                                )
                    else:
                        print(
                            f"Warning: Shape mismatch for {key}: "
                            f"checkpoint {ckpt_shape} vs model {model_shape}, "
                            f"TP size={tp_size}. Skipping this key."
                        )
                        continue

                current_state_dict[key] = value.to(current_state_dict[key].device)

        model.load_state_dict(current_state_dict)
        torch.distributed.barrier()

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        mtq.print_quant_summary(model)

    mtq.fold_weight(model)
    for name, module in model.named_modules():
        if name.endswith("weight_quantizer"):
            assert not module.is_enabled, f"quantizer {name} is still enabled"


class FakeQuantWorker(BaseWorker):
    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        model = self.model_runner.model
        if hasattr(model, "unwrap"):
            model = model.unwrap()
        with disable_compilation(model):
            return super().determine_available_memory()

    def compile_or_warm_up_model(self) -> None:
        if quant_config["quant_cfg"] or quant_config["kv_quant_cfg"]:
            _fakequant_run_prolog_worker(self)
        super().compile_or_warm_up_model()
