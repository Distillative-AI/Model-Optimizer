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

"""Code that export quantized Hugging Face models for deployment."""

import collections.abc
import json
import re
import tempfile
import warnings
from builtins import ValueError
from collections import defaultdict
from pathlib import Path
from typing import Any

import diffusers
import torch
import torch.nn as nn
from diffusers import DiffusionPipeline, ModelMixin
from safetensors.torch import save_file
from torch.distributed.fsdp import FSDPModule

from modelopt.torch.quantization import set_quantizer_by_cfg_context
from modelopt.torch.quantization.nn import SequentialQuantizer, TensorQuantizer
from modelopt.torch.quantization.qtensor import NVFP4QTensor
from modelopt.torch.quantization.utils import fsdp2_aware_weight_update, quantizer_attr_names

from .convert_hf_config import convert_hf_quant_config_format
from .layer_utils import (
    get_expert_linear_names,
    get_experts_list,
    is_layernorm,
    is_moe,
    is_quantlinear,
    set_expert_quantizer_amax,
)
from .model_config import (
    KV_CACHE_FP8,
    KV_CACHE_NVFP4,
    KV_CACHE_NVFP4_AFFINE,
    QUANTIZATION_FP8,
    QUANTIZATION_FP8_PB_REAL,
    QUANTIZATION_FP8_PC_PT,
    QUANTIZATION_NONE,
    QUANTIZATION_NVFP4,
    QUANTIZATION_NVFP4_AWQ,
    QUANTIZATION_W4A8_AWQ,
    QUANTIZATION_W4A8_NVFP4_FP8,
)
from .model_utils import get_language_model_from_vl, is_multimodal_model
from .plugins import export_spec_ckpt_config, export_spec_ckpt_state_dict, spec_opt_only
from .quant_utils import (
    fuse_prequant_layernorm,
    fuse_prequant_to_linear,
    get_activation_scaling_factor,
    get_quant_config,
    get_quantization_format,
    get_weight_block_size,
    get_weight_scaling_factor,
    get_weight_scaling_factor_2,
    maybe_transpose_expert_weight_dimensions,
    postprocess_state_dict,
    preprocess_linear_fusion,
    to_quantized_weight,
)

__all__ = ["export_hf_checkpoint"]


from contextlib import contextmanager


@contextmanager
def _hide_quantizers_from_state_dict(model: nn.Module):
    """Context manager that temporarily removes quantizer modules from the model.

    This allows save_pretrained to save the model without quantizer buffers like _amax.
    The quantizers are restored after exiting the context.

    Args:
        model: The model with quantizers to temporarily hide.

    Yields:
        None - the model can be saved within the context.
    """
    # Store references to quantizers that we'll temporarily remove
    quantizer_backup: dict[str, dict[str, nn.Module]] = {}

    for name, module in model.named_modules():
        if is_quantlinear(module):
            backup = {}
            for attr in ["weight_quantizer", "input_quantizer", "output_quantizer"]:
                if hasattr(module, attr):
                    backup[attr] = getattr(module, attr)
                    delattr(module, attr)
            if backup:
                quantizer_backup[name] = backup

    try:
        yield
    finally:
        # Restore quantizers
        for name, backup in quantizer_backup.items():
            module = model.get_submodule(name)
            for attr, quantizer in backup.items():
                setattr(module, attr, quantizer)


def _is_enabled_quantizer(quantizer):
    if hasattr(quantizer, "is_enabled") and quantizer.is_enabled:
        return True

    if isinstance(quantizer, SequentialQuantizer):
        return any(q.is_enabled for q in quantizer)

    return False


def requantize_resmooth_fused_llm_layers(model: torch.nn.Module):
    """Group modules that take the same input and register shared parameters in module."""
    # TODO: Handle DBRX MoE
    input_to_linear = defaultdict(list)
    output_to_layernorm = defaultdict(None)
    quantization_format = get_quantization_format(model)

    def _input_hook(module, input, output):
        """Update dictionary with list of all modules that share the same input."""
        # TODO: Handle DBRX MoE case
        input_to_linear[input[0]].append(module)

    def _output_hook(module, input, output):
        """Update dictionary with mapping of layernorms and their outputs."""
        output_to_layernorm[output] = module

    handles = []
    model_type = type(model).__name__.lower()

    fused_linears = {}
    module_names = set()

    # Fuse pre_quant_scale to the linear weights if possible
    if quantization_format is not None and "nvfp4_awq" in quantization_format.lower():
        fuse_prequant_to_linear(model)

    for name, module in model.named_modules():
        module_names.add(name)

        # For MoE models update pre_quant_scale to average pre_quant_scale amongst experts
        if is_moe(module) and ("awq" in quantization_format):
            # update_experts_avg_prequant_scale(module)
            grouped_experts = get_experts_list(module, model_type)
            for modules in grouped_experts:
                with fsdp2_aware_weight_update(model, modules):
                    preprocess_linear_fusion(modules, resmooth_only=True)

        # Attach hook to layernorm modules that need to be fused
        if is_layernorm(module):
            module.name = name
            handle = module.register_forward_hook(_output_hook)
            handles.append(handle)
        elif is_quantlinear(module) and (
            _is_enabled_quantizer(module.input_quantizer)
            or _is_enabled_quantizer(module.weight_quantizer)
        ):
            module.name = name
            handle = module.register_forward_hook(_input_hook)
            handles.append(handle)

    with torch.no_grad():
        fake_input = torch.ones([1, 2], dtype=torch.long).to(model.device)
        decoder_fake_input = fake_input

        # Check if this is a VL model that needs special input handling
        is_vl_model = is_multimodal_model(model)

        if model_type.startswith("whisper"):
            # For Whisper models, we need to pass a fake input with the specific sequence length
            from transformers import AutoFeatureExtractor

            feature_extractor = AutoFeatureExtractor.from_pretrained(model.name_or_path)
            fake_input = torch.ones(
                [1, model.config.num_mel_bins, feature_extractor.nb_max_frames], dtype=model.dtype
            ).to(model.device)

        # Run forward pass so that all modules sharing the same input are collected using forward hook.

        with set_quantizer_by_cfg_context(model, {"*": {"enable": False}}):
            if getattr(model.config, "is_encoder_decoder", False):
                # For encoder-decoder models, we need to pass both the encoder and decoder input ids
                model(fake_input, decoder_input_ids=decoder_fake_input)
            elif is_vl_model and "nemotron" in model_type:
                # For Nemotron VL models, try to run optimization on just the language model part
                language_model_lineage = get_language_model_from_vl(model)

                if language_model_lineage is not None:
                    # Run optimization on just the language model with the same input format as regular LLMs
                    # Use the same fake_input tensor that regular LLMs use
                    language_model = language_model_lineage[-1]
                    print(
                        f"Running optimization on language model with fake_input shape: {fake_input.shape}"
                    )
                    language_model(fake_input)
                else:
                    raise ValueError(
                        f"Cannot extract language_model from Nemotron VL model (type: {model_type}). "
                        "This is required for requantization/resmoothing optimization. "
                        "Please ensure the model architecture is supported or file an issue."
                    )
            else:
                model(fake_input)

        for handle in handles:
            handle.remove()

    for tensor, modules in input_to_linear.items():
        quantization_format = get_quantization_format(modules[0])
        if len(modules) > 1 and quantization_format not in [
            QUANTIZATION_FP8,
            QUANTIZATION_NONE,
            QUANTIZATION_FP8_PB_REAL,
        ]:
            # Fuse modules that have the same input
            with fsdp2_aware_weight_update(model, modules):
                preprocess_linear_fusion(modules)
            fused_linears[modules[0].name] = [module.name for module in modules]

        # Fuse layernorms
        if (
            quantization_format is not QUANTIZATION_NONE
            and "awq" in quantization_format
            and tensor in output_to_layernorm
        ):
            # Pre quant scale of modules is already updated to avg_pre_quant_scale
            with fsdp2_aware_weight_update(model, output_to_layernorm[tensor]):
                fuse_prequant_layernorm(output_to_layernorm[tensor], modules)

    # The dummy forward may not be able to activate all the experts.
    # Process experts by naming rules like experts.0, experts.1, etc.
    for name, modules_fused in fused_linears.items():
        if re.search(r"experts?\.\d+", name):
            expert_id = 0
            while True:
                new_expert_name = re.sub(r"(experts?\.)\d+", rf"\g<1>{expert_id}", name, count=1)
                if new_expert_name in fused_linears:
                    expert_id += 1
                    continue
                if new_expert_name not in module_names:
                    break

                new_expert_modules = []
                for name_fused in modules_fused:
                    new_expert_name = re.sub(r"(experts?\.)\d+", rf"\g<1>{expert_id}", name_fused)
                    assert new_expert_name in module_names
                    new_expert_modules.append(model.get_submodule(new_expert_name))

                with fsdp2_aware_weight_update(model, new_expert_modules):
                    preprocess_linear_fusion(new_expert_modules)

                expert_id += 1


def _export_quantized_weight(
    sub_module: nn.Module, dtype: torch.dtype, weight_name: str = "weight"
):
    """For the given weight attr of the sub_module, export the quantization info of it.

    The export includes converting weight tensor to correct quantized values and quantized dtype,
    and registering scaling factors.
    """
    quantization_format = get_quantization_format(sub_module)
    if quantization_format == QUANTIZATION_NONE:
        return

    block_size = get_weight_block_size(sub_module, weight_name)
    quantizer_attrs = quantizer_attr_names(weight_name)
    weight: nn.Parameter = getattr(sub_module, weight_name)
    weight_quantizer: TensorQuantizer | SequentialQuantizer = getattr(
        sub_module, quantizer_attrs.weight_quantizer
    )
    input_quantizer: TensorQuantizer | SequentialQuantizer | None = getattr(
        sub_module, quantizer_attrs.input_quantizer, None
    )
    output_quantizer: TensorQuantizer | SequentialQuantizer | None = getattr(
        sub_module, quantizer_attrs.output_quantizer, None
    )

    if quantization_format == QUANTIZATION_FP8:
        # Convert amax to float32
        weight_quantizer._amax = weight_quantizer._amax.to(torch.float32)

        if weight_quantizer._amax.dim() == 1:
            # Per-tensor amax
            weight_scaling_factor = torch.tensor(
                weight_quantizer.amax.item() / weight_quantizer.maxbound
            )
        else:
            # Per-channel amax
            weight_scaling_factor = torch.tensor(weight_quantizer.amax / weight_quantizer.maxbound)

        sub_module.register_buffer(
            quantizer_attrs.weight_scale,
            weight_scaling_factor,
        )

        if hasattr(input_quantizer, "_amax"):
            assert input_quantizer is not None
            input_quantizer._amax = input_quantizer._amax.to(torch.float32)

            sub_module.register_buffer(
                quantizer_attrs.input_scale,
                get_activation_scaling_factor(
                    sub_module, input_quantizer_name=quantizer_attrs.input_quantizer
                ).squeeze(),
            )

        if hasattr(output_quantizer, "_amax"):
            assert output_quantizer is not None
            output_quantizer._amax = output_quantizer._amax.to(torch.float32)
    else:
        # Register weight_scale and input_scale
        if quantization_format == QUANTIZATION_FP8_PB_REAL:
            sub_module.register_buffer(
                quantizer_attrs.weight_scale,
                weight_quantizer._scale.to(torch.float32),
            )
            del weight_quantizer._scale
        else:
            sub_module.register_buffer(
                quantizer_attrs.weight_scale, get_weight_scaling_factor(sub_module, weight_name)
            )

        if (
            input_quantizer is not None
            and "disabled" not in repr(input_quantizer)
            and input_quantizer.amax is not None
        ):
            sub_module.register_buffer(
                quantizer_attrs.input_scale,
                get_activation_scaling_factor(
                    sub_module, input_quantizer_name=quantizer_attrs.input_quantizer
                ).squeeze(),
            )

    if quantization_format in [
        QUANTIZATION_NVFP4_AWQ,
        QUANTIZATION_NVFP4,
        QUANTIZATION_W4A8_AWQ,
        QUANTIZATION_W4A8_NVFP4_FP8,
    ]:
        # Register weight_scale_2
        sub_module.register_buffer(
            quantizer_attrs.weight_scale_2,
            get_weight_scaling_factor_2(sub_module, weight_name).squeeze(),
        )

    weight_scale: torch.Tensor | None = getattr(sub_module, quantizer_attrs.weight_scale, None)
    weight_scale_2: torch.Tensor | None = getattr(sub_module, quantizer_attrs.weight_scale_2, None)

    # Transpose weight for bmm-style expert quantization (llama4, gpt-oss)
    # Check if this is a BMM-style expert weight that needs transposition
    is_bmm_expert_weight = weight.dim() == 3 and any(
        expert_type in type(sub_module).__name__
        for expert_type in ["Llama4TextExperts", "GptOssExperts"]
    )

    if quantization_format in [QUANTIZATION_NVFP4, QUANTIZATION_NVFP4_AWQ]:
        # Transpose weight from (num_experts, input_dim, output_dim) to (num_experts, output_dim, input_dim)
        # for NVFP4 quantization functions that expect input_dim as the last dimension for block quantization
        weight, _ = maybe_transpose_expert_weight_dimensions(
            weight, is_bmm_expert_weight=is_bmm_expert_weight
        )
        weight_scale = NVFP4QTensor.get_weights_scaling_factor(
            weight,
            block_size=block_size,
            weights_scaling_factor_2=weight_scale_2,
        )[0]

        quantized_weight = to_quantized_weight(
            weight.to(dtype),
            weight_scale,
            quantization_format,
            weight_scale_2,
            block_size,
        )

        quantized_weight, weight_scale = maybe_transpose_expert_weight_dimensions(
            quantized_weight, weight_scale, is_bmm_expert_weight=is_bmm_expert_weight
        )
    elif quantization_format == QUANTIZATION_FP8_PC_PT and is_bmm_expert_weight:
        # For FP8_PC_PT with BMM-style experts, transpose only the weight (not weight_scale)
        weight, _ = maybe_transpose_expert_weight_dimensions(
            weight, is_bmm_expert_weight=is_bmm_expert_weight
        )

        quantized_weight = to_quantized_weight(
            weight.to(dtype),
            weight_scale,
            quantization_format,
            weight_scale_2,
            block_size,
        )

        # Transpose back to original BMM format
        quantized_weight, _ = maybe_transpose_expert_weight_dimensions(
            quantized_weight, is_bmm_expert_weight=is_bmm_expert_weight
        )
    else:
        quantized_weight = to_quantized_weight(
            weight.to(dtype),
            weight_scale,
            quantization_format,
            weight_scale_2,
            block_size,
        )

    setattr(sub_module, weight_name, nn.Parameter(quantized_weight, requires_grad=False))

    # Register the corrected weight_scale as a buffer
    if weight_scale is not None:
        sub_module.register_buffer(quantizer_attrs.weight_scale, weight_scale)


def _process_quantized_modules(
    model: nn.Module,
    dtype: torch.dtype,
    is_modelopt_qlora: bool = False,
) -> None:
    """Process all quantized modules in model, export weights in-place.

    This function iterates through all modules in the model and exports quantized weights
    for modules that have quantization enabled. It handles both standard linear layers
    and specialized expert modules (Llama4TextExperts, GptOssExperts).

    Args:
        model: The model containing quantized modules.
        dtype: The data type for weight conversion.
        is_modelopt_qlora: Whether the model is a modelopt-trained QLoRA model.
            If True, modules with base_layer attribute are skipped.
    """
    fsdp_module_to_reshard = None

    for _, sub_module in model.named_modules():
        # Optimization to perform resharding only once per decoder layer to avoid extra communication overhead
        if isinstance(sub_module, FSDPModule):
            # Every time we encounter a new FSDPModule, the previous decoder layer is fully processed.
            # We need to reshard the previous FSDPModule to prevent potential OOM.
            # This hack reduces the number of unshard reshard operations, to avoid unnecessary communication.
            if fsdp_module_to_reshard is not None:
                fsdp_module_to_reshard.reshard()

            fsdp_module_to_reshard = sub_module

        # We skip QuantLoraLinear module for modelopt QLoRA
        if is_modelopt_qlora and (hasattr(sub_module, "base_layer")):
            continue

        if get_quantization_format(sub_module) != QUANTIZATION_NONE:
            if is_quantlinear(sub_module):
                with fsdp2_aware_weight_update(model, sub_module, reshard=False):
                    _export_quantized_weight(sub_module, dtype)
            elif (
                "Llama4TextExperts" in type(sub_module).__name__
                or "GptOssExperts" in type(sub_module).__name__
            ):
                # TODO: consolidate uncalibrated experts handling logic
                # Handle weight quantizers amax values using smart fallback logic
                set_expert_quantizer_amax(
                    modules=sub_module,
                    quantizer_attrs=["gate_up_proj_weight_quantizer", "down_proj_weight_quantizer"],
                )
                # Handle input quantizers amax values using smart fallback logic
                set_expert_quantizer_amax(
                    modules=sub_module,
                    quantizer_attrs=["gate_up_proj_input_quantizer", "down_proj_input_quantizer"],
                )
                # Export the quantized weights
                with fsdp2_aware_weight_update(model, sub_module, reshard=False):
                    for weight_name in ["gate_up_proj", "down_proj"]:
                        _export_quantized_weight(sub_module, dtype, weight_name)


def _export_transformers_checkpoint(
    model: nn.Module, dtype: torch.dtype | None = None, is_modelopt_qlora: bool = False, **kwargs
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Exports the torch model to the packed checkpoint with original HF naming.

    The packed checkpoint will be consumed by the TensorRT-LLM unified converter.

    Args:
        model: the full torch model to export. The actual quantized model may be a submodule.
        dtype: the weights data type to export the unquantized layers or the default model data type if None.
        accelerator: the accelerator instance in case of distributed export setup.

    Returns:
        post_state_dict: Dict containing quantized weights
        quant_config: config information to export hf_quant_cfg.json
    """
    if dtype is None:
        dtype = model.config.torch_dtype
    elif dtype != model.config.torch_dtype:
        warnings.warn(
            f"Model's original dtype ({model.config.torch_dtype}) differs from target dtype "
            f"({dtype}), which may lead to numerical errors."
        )

    accelerator = kwargs.get("accelerator")

    # Handle input quantizers of experts that are not calibrated
    for _, sub_module in model.named_modules():
        if is_moe(sub_module) and hasattr(sub_module, "experts"):
            expert_linear_names = get_expert_linear_names(sub_module)
            for linear_name in expert_linear_names:
                # Handle DBRX experts specifically
                if "QuantDbrxExperts" in type(sub_module.experts).__name__:
                    # For DBRX, experts are in sub_module.experts.mlp and linear layers are ModuleLists
                    experts_mlp = sub_module.experts.mlp
                    if hasattr(experts_mlp, linear_name):
                        linear_modulelist = getattr(experts_mlp, linear_name)
                        if hasattr(linear_modulelist, "__iter__"):
                            set_expert_quantizer_amax(
                                modules=list(linear_modulelist),
                                quantizer_attrs=["input_quantizer"],
                            )
                elif "QuantGptOssExperts" in type(sub_module.experts).__name__:
                    # Handle GPT-OSS experts specifically
                    # GPT-OSS experts use gate_up_proj and down_proj
                    gpt_oss_linear_names = ["gate_up_proj", "down_proj"]
                    for linear_name in gpt_oss_linear_names:
                        if hasattr(sub_module.experts, linear_name):
                            linear_module = getattr(sub_module.experts, linear_name)
                            if hasattr(linear_module, "input_quantizer"):
                                set_expert_quantizer_amax(
                                    modules=[linear_module],
                                    quantizer_attrs=["input_quantizer"],
                                )
                elif isinstance(sub_module.experts, collections.abc.Iterable):
                    # For other MoE models (like Mixtral) with iterable experts
                    try:
                        set_expert_quantizer_amax(
                            modules=[getattr(expert, linear_name) for expert in sub_module.experts],
                            quantizer_attrs=["input_quantizer"],
                        )
                    except AttributeError as e:
                        # Provide more helpful debugging information
                        expert_types = [type(expert).__name__ for expert in sub_module.experts]
                        raise AttributeError(
                            f"Failed to access attribute '{linear_name}' on experts. "
                            f"MoE module type: {type(sub_module).__name__}, "
                            f"Expert types: {expert_types}, "
                            f"Expected linear names: {expert_linear_names}. "
                            f"This suggests the get_expert_linear_names function may need "
                            f"to be updated for this model architecture. "
                            f"Original error: {e}"
                        ) from e
                else:
                    # Unsupported MoE model structure
                    raise NotImplementedError(
                        f"MoE model with experts type '{type(sub_module.experts).__name__}' is not supported in export."
                        f"Please file an issue or add support for this model architecture."
                    )

    # Resmooth and requantize fused layers
    # TODO: Handle mixed precision
    requantize_resmooth_fused_llm_layers(model)

    # Remove all hooks from the model
    try:
        from accelerate.hooks import remove_hook_from_module

        remove_hook_from_module(model, recurse=True)
    except ImportError:
        warnings.warn("accelerate is not installed, hooks will not be removed")

    quant_config = get_quant_config(model, is_modelopt_qlora=is_modelopt_qlora)

    kv_cache_max_bound = 0
    kv_cache_format = quant_config["quantization"]["kv_cache_quant_algo"]

    cache_bound_mapping = {
        KV_CACHE_NVFP4: 6 * 448,
        KV_CACHE_NVFP4_AFFINE: 6 * 448,
        KV_CACHE_FP8: 448,
    }

    # Only update kv_cache_max_bound if a quantization is applied.
    if kv_cache_format != QUANTIZATION_NONE:
        kv_cache_max_bound = cache_bound_mapping.get(kv_cache_format)

    # Process all quantized modules and export weights
    _process_quantized_modules(model, dtype, is_modelopt_qlora)

    if accelerator is not None:
        # Gather state_dict from all ranks
        quantized_state_dict = accelerator.get_state_dict(model)
    else:
        quantized_state_dict = model.state_dict()

    quantized_state_dict = postprocess_state_dict(
        quantized_state_dict, kv_cache_max_bound, kv_cache_format, is_modelopt_qlora
    )

    return quantized_state_dict, quant_config


def _generate_diffusion_dummy_inputs(
    model: nn.Module, device: torch.device, dtype: torch.dtype
) -> dict[str, torch.Tensor] | None:
    """Generate dummy inputs for diffusion model forward pass.

    Different diffusion models have very different input formats:
    - DiTTransformer2DModel: 4D hidden_states + class_labels
    - FluxTransformer2DModel: 3D hidden_states + encoder_hidden_states + img_ids + txt_ids + pooled_projections
    - SD3Transformer2DModel: 4D hidden_states + encoder_hidden_states + pooled_projections
    - UNet2DConditionModel: 4D sample + timestep + encoder_hidden_states

    Args:
        model: The diffusion model component.
        device: Device to create tensors on.
        dtype: Data type for tensors.

    Returns:
        Dictionary of dummy inputs, or None if model type is not supported.
    """
    model_class_name = type(model).__name__
    batch_size = 1

    # Try to import specific model classes for isinstance checks
    try:
        from diffusers.models.transformers import FluxTransformer2DModel

        is_flux = isinstance(model, FluxTransformer2DModel)
    except ImportError:
        is_flux = "flux" in model_class_name.lower()

    try:
        from diffusers.models.transformers import SD3Transformer2DModel

        is_sd3 = isinstance(model, SD3Transformer2DModel)
    except ImportError:
        is_sd3 = "sd3" in model_class_name.lower()

    try:
        from diffusers.models.transformers import DiTTransformer2DModel

        is_dit = isinstance(model, DiTTransformer2DModel)
    except ImportError:
        is_dit = model_class_name == "DiTTransformer2DModel"

    try:
        from diffusers.models.unets import UNet2DConditionModel

        is_unet = isinstance(model, UNet2DConditionModel)
    except ImportError:
        is_unet = "unet" in model_class_name.lower()

    cfg = getattr(model, "config", None)

    if is_flux:
        # FluxTransformer2DModel: 3D hidden_states (batch, seq_len, in_channels)
        # Requires: hidden_states, encoder_hidden_states, pooled_projections, timestep, img_ids, txt_ids
        in_channels = getattr(cfg, "in_channels", 64)
        joint_attention_dim = getattr(cfg, "joint_attention_dim", 4096)
        pooled_projection_dim = getattr(cfg, "pooled_projection_dim", 768)
        guidance_embeds = getattr(cfg, "guidance_embeds", False)

        # Use small dimensions for dummy forward
        img_seq_len = 16  # 4x4 latent grid
        text_seq_len = 8

        dummy_inputs = {
            "hidden_states": torch.randn(
                batch_size, img_seq_len, in_channels, device=device, dtype=dtype
            ),
            "encoder_hidden_states": torch.randn(
                batch_size, text_seq_len, joint_attention_dim, device=device, dtype=dtype
            ),
            "pooled_projections": torch.randn(
                batch_size, pooled_projection_dim, device=device, dtype=dtype
            ),
            "timestep": torch.tensor([0.5], device=device, dtype=dtype).expand(batch_size),
            "img_ids": torch.zeros(img_seq_len, 3, device=device, dtype=torch.float32),
            "txt_ids": torch.zeros(text_seq_len, 3, device=device, dtype=torch.float32),
            "return_dict": False,
        }
        if guidance_embeds:
            dummy_inputs["guidance"] = torch.tensor([3.5], device=device, dtype=torch.float32)
        return dummy_inputs

    elif is_sd3:
        # SD3Transformer2DModel: 4D hidden_states (batch, channels, height, width)
        # Requires: hidden_states, encoder_hidden_states, pooled_projections, timestep
        in_channels = getattr(cfg, "in_channels", 16)
        sample_size = getattr(cfg, "sample_size", 128)
        joint_attention_dim = getattr(cfg, "joint_attention_dim", 4096)
        pooled_projection_dim = getattr(cfg, "pooled_projection_dim", 2048)

        # Use smaller sample size for speed
        test_size = min(sample_size, 32)
        text_seq_len = 8

        return {
            "hidden_states": torch.randn(
                batch_size, in_channels, test_size, test_size, device=device, dtype=dtype
            ),
            "encoder_hidden_states": torch.randn(
                batch_size, text_seq_len, joint_attention_dim, device=device, dtype=dtype
            ),
            "pooled_projections": torch.randn(
                batch_size, pooled_projection_dim, device=device, dtype=dtype
            ),
            "timestep": torch.randint(0, 1000, (batch_size,), device=device),
            "return_dict": False,
        }

    elif is_dit:
        # DiTTransformer2DModel: 4D hidden_states (batch, in_channels, height, width)
        # Requires: hidden_states, timestep, class_labels
        in_channels = getattr(cfg, "in_channels", 4)
        sample_size = getattr(cfg, "sample_size", 32)
        num_embeds_ada_norm = getattr(cfg, "num_embeds_ada_norm", 1000)

        # Use smaller sample size for speed
        test_size = min(sample_size, 16)

        return {
            "hidden_states": torch.randn(
                batch_size, in_channels, test_size, test_size, device=device, dtype=dtype
            ),
            "timestep": torch.randint(0, num_embeds_ada_norm, (batch_size,), device=device),
            "class_labels": torch.randint(0, num_embeds_ada_norm, (batch_size,), device=device),
            "return_dict": False,
        }

    elif is_unet:
        # UNet2DConditionModel: 4D sample (batch, in_channels, height, width)
        # Requires: sample, timestep, encoder_hidden_states
        in_channels = getattr(cfg, "in_channels", 4)
        sample_size = getattr(cfg, "sample_size", 64)
        cross_attention_dim = getattr(cfg, "cross_attention_dim", 768)

        # Use smaller sample size for speed
        test_size = min(sample_size, 32)
        text_seq_len = 8

        dummy_inputs = {
            "sample": torch.randn(
                batch_size, in_channels, test_size, test_size, device=device, dtype=dtype
            ),
            "timestep": torch.randint(0, 1000, (batch_size,), device=device),
            "encoder_hidden_states": torch.randn(
                batch_size, text_seq_len, cross_attention_dim, device=device, dtype=dtype
            ),
            "return_dict": False,
        }

        # Handle SDXL additional conditioning
        if getattr(cfg, "addition_embed_type", None) == "text_time":
            # SDXL requires text_embeds and time_ids
            add_embed_dim = getattr(cfg, "projection_class_embeddings_input_dim", 2816)
            dummy_inputs["added_cond_kwargs"] = {
                "text_embeds": torch.randn(
                    batch_size, add_embed_dim - 6 * 256, device=device, dtype=dtype
                ),
                "time_ids": torch.randn(batch_size, 6, device=device, dtype=dtype),
            }
        return dummy_inputs

    # Try generic transformer handling for other model types
    # Check if model has common transformer attributes
    elif cfg is not None:
        # Many transformers use 4D hidden_states with in_channels and sample_size
        if hasattr(cfg, "in_channels") and hasattr(cfg, "sample_size"):
            in_channels = cfg.in_channels
            sample_size = cfg.sample_size
            test_size = min(sample_size, 32)

            dummy_inputs = {
                "hidden_states": torch.randn(
                    batch_size, in_channels, test_size, test_size, device=device, dtype=dtype
                ),
                "timestep": torch.randint(0, 1000, (batch_size,), device=device),
                "return_dict": False,
            }

            # Add encoder_hidden_states if model has cross attention
            if hasattr(cfg, "joint_attention_dim"):
                text_seq_len = 8
                dummy_inputs["encoder_hidden_states"] = torch.randn(
                    batch_size, text_seq_len, cfg.joint_attention_dim, device=device, dtype=dtype
                )
                if hasattr(cfg, "pooled_projection_dim"):
                    dummy_inputs["pooled_projections"] = torch.randn(
                        batch_size, cfg.pooled_projection_dim, device=device, dtype=dtype
                    )
            elif hasattr(cfg, "cross_attention_dim"):
                text_seq_len = 8
                dummy_inputs["encoder_hidden_states"] = torch.randn(
                    batch_size, text_seq_len, cfg.cross_attention_dim, device=device, dtype=dtype
                )

            return dummy_inputs

    return None


def _is_qkv_projection(module_name: str) -> bool:
    """Check if a module name corresponds to a QKV projection layer.

    In diffusers, QKV projections typically have names like:
    - to_q, to_k, to_v (most common in diffusers attention)
    - q_proj, k_proj, v_proj
    - query, key, value
    - add_q_proj, add_k_proj, add_v_proj (for additional attention in some models)

    We exclude:
    - norm*.linear (AdaLayerNorm modulation layers)
    - proj_out, proj_mlp (output projections)
    - ff.*, mlp.* (feed-forward layers)
    - to_out (output projection)

    Args:
        module_name: The full module name path.

    Returns:
        True if this is a QKV projection layer.
    """
    # Get the last component of the module name
    name_parts = module_name.split(".")
    last_part = name_parts[-1] if name_parts else ""
    second_last = name_parts[-2] if len(name_parts) >= 2 else ""

    # QKV projection patterns (positive matches)
    qkv_patterns = [
        "to_q",
        "to_k",
        "to_v",
        "q_proj",
        "k_proj",
        "v_proj",
        "query",
        "key",
        "value",
        "add_q_proj",
        "add_k_proj",
        "add_v_proj",
        "to_added_q",
        "to_added_k",
        "to_added_v",
    ]

    # Check if the last part matches any QKV pattern
    if last_part in qkv_patterns:
        return True

    # Also check second-to-last for cases like "attn.to_q.weight"
    return second_last in qkv_patterns


def _get_qkv_group_key(module_name: str) -> str:
    """Extract the parent attention block path and QKV type for grouping.

    QKV projections should only be fused within the same attention block AND
    for the same type of attention (main vs added/cross).

    Examples:
        - 'transformer_blocks.0.attn.to_q' -> 'transformer_blocks.0.attn.main'
        - 'transformer_blocks.0.attn.to_k' -> 'transformer_blocks.0.attn.main'
        - 'transformer_blocks.5.attn.add_q_proj' -> 'transformer_blocks.5.attn.add'
        - 'transformer_blocks.5.attn.add_k_proj' -> 'transformer_blocks.5.attn.add'

    Args:
        module_name: The full module name path.

    Returns:
        A string key representing the attention block and QKV type for grouping.
    """
    name_parts = module_name.split(".")
    last_part = name_parts[-1] if name_parts else ""

    # Determine if this is "main" QKV or "added" QKV (for cross-attention in some models)
    added_patterns = [
        "add_q_proj",
        "add_k_proj",
        "add_v_proj",
        "to_added_q",
        "to_added_k",
        "to_added_v",
    ]
    qkv_type = "add" if last_part in added_patterns else "main"

    # Find the parent attention block by removing the QKV projection name
    # e.g., 'transformer_blocks.0.attn.to_q' -> 'transformer_blocks.0.attn'
    parent_parts = name_parts[:-1]
    parent_path = ".".join(parent_parts) if parent_parts else ""

    return f"{parent_path}.{qkv_type}"


def _fuse_qkv_linears_diffusion(model: nn.Module) -> None:
    """Fuse QKV linear layers that share the same input for diffusion models.

    This function uses forward hooks to dynamically identify linear modules that
    share the same input tensor (e.g., q_proj, k_proj, v_proj in attention).
    For these modules, it unifies their input and weight amax values.

    Note: This is a simplified version for diffusion models that:
    - Handles QKV fusion (shared input detection)
    - Filters to only fuse actual QKV projection layers (not AdaLN, FFN, etc.)
    - Skips pre_quant_scale handling (TODO for future)
    - Skips FFN fusion with layernorm (TODO for future)

    Args:
        model: The diffusion model component (e.g., transformer, unet).
    """
    from modelopt.torch.quantization import set_quantizer_by_cfg_context

    input_to_linear: dict[int, list[nn.Module]] = defaultdict(list)
    quantization_format = get_quantization_format(model)

    if quantization_format == QUANTIZATION_NONE:
        return

    def _input_hook(module, input, output):
        """Collect modules that share the same input tensor."""
        if len(input) > 0 and isinstance(input[0], torch.Tensor):
            # Use tensor data pointer as key to identify same tensor
            input_to_linear[input[0].data_ptr()].append(module)

    handles = []

    # Register hooks on all quantized linear modules
    for name, module in model.named_modules():
        if is_quantlinear(module) and (
            _is_enabled_quantizer(module.input_quantizer)
            or _is_enabled_quantizer(module.weight_quantizer)
        ):
            module.name = name
            handle = module.register_forward_hook(_input_hook)
            handles.append(handle)

    if not handles:
        print("No quantized linear modules found for QKV fusion.")
        return

    # Run dummy forward pass to collect modules sharing same input
    try:
        with torch.no_grad():
            device = next(model.parameters()).device
            dtype = next(model.parameters()).dtype

            # Disable quantizers during dummy forward to avoid numerical issues
            with set_quantizer_by_cfg_context(model, {"*": {"enable": False}}):
                # Generate appropriate dummy inputs based on model type
                dummy_inputs = _generate_diffusion_dummy_inputs(model, device, dtype)

                if dummy_inputs is None:
                    model_class_name = type(model).__name__
                    print(f"Warning: Unknown model type '{model_class_name}', skipping QKV fusion.")
                    for handle in handles:
                        handle.remove()
                    return

                # Run forward pass with dummy inputs
                model(**dummy_inputs)

    except Exception as e:
        print(f"Warning: Failed to run dummy forward for QKV fusion: {e}")
        print("Skipping QKV fusion. Quantization may still work but amax values won't be unified.")
        for handle in handles:
            handle.remove()
        return

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Process modules that share the same input
    fused_count = 0
    for modules in input_to_linear.values():
        if len(modules) > 1 and quantization_format not in [
            QUANTIZATION_FP8,
            QUANTIZATION_NONE,
            QUANTIZATION_FP8_PB_REAL,
        ]:
            # Filter to only include QKV projection layers
            qkv_modules = [m for m in modules if _is_qkv_projection(getattr(m, "name", ""))]

            if len(qkv_modules) > 1:
                # Group QKV modules by their parent attention block
                # This ensures we only fuse Q, K, V within the same attention layer
                qkv_groups: dict[str, list[nn.Module]] = defaultdict(list)
                for m in qkv_modules:
                    group_key = _get_qkv_group_key(getattr(m, "name", ""))
                    qkv_groups[group_key].append(m)

                # Fuse each group separately
                for group_key, group_modules in qkv_groups.items():
                    if len(group_modules) > 1:
                        # These are QKV modules from the same attention block - fuse their amax values
                        preprocess_linear_fusion(group_modules, resmooth_only=False)
                        fused_count += 1
                        module_names = [getattr(m, "name", "unknown") for m in group_modules]
                        print(f"  Fused QKV group: {module_names}")

    if fused_count > 0:
        print(f"Fused {fused_count} QKV group(s) for unified amax values.")
    else:
        print("No QKV groups found to fuse.")


def _get_diffusers_components(
    model: DiffusionPipeline,
    components: list[str] | None = None,
) -> dict[str, Any]:
    """Get all exportable components from a diffusers pipeline.

    This function extracts all components from a DiffusionPipeline including
    nn.Module models, tokenizers, schedulers, feature extractors, etc.

    Args:
        model: The diffusers pipeline.
        components: Optional list of component names to filter. If None, all
            components are returned.

    Returns:
        Dictionary mapping component names to their instances (can be nn.Module,
        tokenizers, schedulers, etc.).
    """
    if isinstance(model, DiffusionPipeline):
        # Get all components from the pipeline
        all_components = {name: comp for name, comp in model.components.items() if comp is not None}

        # If specific components requested, filter to only those
        if components is not None:
            filtered = {name: comp for name, comp in all_components.items() if name in components}
            # Warn about requested components that don't exist
            missing = set(components) - set(filtered.keys())
            if missing:
                warnings.warn(f"Requested components not found in pipeline: {missing}")
            return filtered

        return all_components
    else:
        raise TypeError(f"Expected DiffusionPipeline for now, got {type(model).__name__}")


def _has_quantized_modules(model: nn.Module) -> bool:
    """Check if a model has any quantized modules.

    Args:
        model: The model to check.

    Returns:
        True if the model contains quantized modules, False otherwise.
    """
    return any(
        get_quantization_format(sub_module) != QUANTIZATION_NONE
        for _, sub_module in model.named_modules()
    )


def _infer_dtype_from_model(model: nn.Module) -> torch.dtype:
    """Infer the dtype from a model's parameters.

    Args:
        model: The model to infer dtype from.

    Returns:
        The dtype of the model's parameters, defaulting to float16 if no parameters found.
    """
    for param in model.parameters():
        return param.dtype
    return torch.float16


def _export_diffusers_checkpoint(
    pipe: DiffusionPipeline | ModelMixin,
    dtype: torch.dtype | None,
    export_dir: Path,
    components: list[str] | None,
    max_shard_size: int | str = "10GB",
) -> None:
    """Internal: Export Diffusers model/pipeline checkpoint.

    This function handles the export of diffusers models, including
    DiffusionPipeline and individual ModelMixin components. It exports all
    components including nn.Module models, tokenizers, schedulers, etc.

    Args:
        pipe: The diffusers model or pipeline to export.
        dtype: The data type for weight conversion. If None, will be inferred from model.
        export_dir: The directory to save the exported checkpoint.
        components: Optional list of component names to export. Only used for pipelines.
            If None, all components are exported.
        max_shard_size: Maximum size of each shard file. If the model exceeds this size,
            it will be sharded into multiple files and a .safetensors.index.json will be
            created. Use smaller values like "5GB" or "2GB" to force sharding.
    """
    export_dir = Path(export_dir)

    # Step 1: Get all pipeline components (nn.Module, tokenizers, schedulers, etc.)
    all_components = _get_diffusers_components(pipe, components)

    if not all_components:
        warnings.warn("No exportable components found in the model.")
        return

    # Separate nn.Module components for quantization-aware export
    module_components = {
        name: comp for name, comp in all_components.items() if isinstance(comp, nn.Module)
    }

    # Step 3: Export each nn.Module component with quantization handling
    for component_name, component in module_components.items():
        is_quantized = _has_quantized_modules(component)
        status = "quantized" if is_quantized else "non-quantized"
        print(f"Exporting component: {component_name} ({status})")

        # Determine component export directory
        # For pipelines, each component goes in a subfolder
        if isinstance(pipe, DiffusionPipeline):
            component_export_dir = export_dir / component_name
        else:
            component_export_dir = export_dir

        component_export_dir.mkdir(parents=True, exist_ok=True)

        # Infer dtype if not provided
        component_dtype = dtype if dtype is not None else _infer_dtype_from_model(component)

        if is_quantized:
            # Step 3.5: Fuse QKV linears that share the same input (unify amax values)
            # This is similar to requantize_resmooth_fused_llm_layers but simplified for diffusion
            # TODO: Add pre_quant_scale handling and FFN fusion for AWQ-style quantization
            print(f"  Running QKV fusion for {component_name}...")
            _fuse_qkv_linears_diffusion(component)

            # Step 4: Process quantized modules (convert weights, register scales)
            _process_quantized_modules(component, component_dtype, is_modelopt_qlora=False)

            # Step 5: Build quantization config
            quant_config = get_quant_config(component, is_modelopt_qlora=False)

            # Step 6: Save the component
            # Note: diffusers ModelMixin.save_pretrained does NOT accept state_dict parameter
            # (unlike transformers), so we use a context manager to temporarily hide quantizers
            # from the state dict during save. This avoids saving quantizer buffers like _amax.
            with _hide_quantizers_from_state_dict(component):
                component.save_pretrained(component_export_dir, max_shard_size=max_shard_size)

            # Step 7: Update config.json with quantization info
            if quant_config is not None:
                hf_quant_config = convert_hf_quant_config_format(quant_config)

                config_path = component_export_dir / "config.json"
                if config_path.exists():
                    with open(config_path) as file:
                        config_data = json.load(file)
                    config_data["quantization_config"] = hf_quant_config
                    with open(config_path, "w") as file:
                        json.dump(config_data, file, indent=4)
        else:
            # Non-quantized component: just save as-is
            component.save_pretrained(component_export_dir, max_shard_size=max_shard_size)

        print(f"  Saved to: {component_export_dir}")

    # Step 4: Export non-nn.Module components (tokenizers, schedulers, feature extractors, etc.)
    if isinstance(pipe, DiffusionPipeline):
        for component_name, component in all_components.items():
            # Skip nn.Module components (already handled above)
            if isinstance(component, nn.Module):
                continue

            component_export_dir = export_dir / component_name
            component_export_dir.mkdir(parents=True, exist_ok=True)

            print(f"Exporting component: {component_name} ({type(component).__name__})")

            # Handle different component types
            if hasattr(component, "save_pretrained"):
                # Tokenizers, feature extractors, image processors
                component.save_pretrained(component_export_dir)
            elif hasattr(component, "save_config"):
                # Schedulers
                component.save_config(component_export_dir)
            else:
                warnings.warn(
                    f"Component '{component_name}' of type {type(component).__name__} "
                    "does not have save_pretrained or save_config method. Skipping."
                )
                continue

            print(f"  Saved to: {component_export_dir}")

    # Step 5: For pipelines, also save the model_index.json
    if isinstance(pipe, DiffusionPipeline):
        model_index_path = export_dir / "model_index.json"
        if hasattr(pipe, "config") and pipe.config is not None:
            # Save a simplified model_index.json that points to the exported components
            model_index = {
                "_class_name": type(pipe).__name__,
                "_diffusers_version": diffusers.__version__,
            }
            # Add component class names for all components
            # Use the base library name (e.g., "diffusers", "transformers") instead of
            # the full module path, as expected by diffusers pipeline loading
            for name, comp in all_components.items():
                module = type(comp).__module__
                # Extract base library name (first part of module path)
                library = module.split(".")[0]
                model_index[name] = [library, type(comp).__name__]

            with open(model_index_path, "w") as file:
                json.dump(model_index, file, indent=4)

    print(f"Export complete. Saved to: {export_dir}")


def export_hf_checkpoint(
    model: nn.Module | DiffusionPipeline,
    dtype: torch.dtype | None = None,
    export_dir: Path | str = tempfile.gettempdir(),
    save_modelopt_state: bool = False,
    components: list[str] | None = None,
):
    """Export quantized HuggingFace model checkpoint (transformers or diffusers).

    This function automatically detects whether the model is from transformers
    or diffusers and applies the appropriate export logic.

    Args:
        model: The full torch model to export. The actual quantized model may be a submodule.
            Supports both transformers models (e.g., LlamaForCausalLM) and diffusers
            models/pipelines (e.g., StableDiffusionPipeline, UNet2DConditionModel).
        dtype: The weights data type to export the unquantized layers or the default
            model data type if None.
        export_dir: The target export path.
        save_modelopt_state: Whether to save the modelopt state_dict.
        components: Only used for diffusers pipelines. Optional list of component names
            to export. If None, all quantized components are exported.
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(model, (DiffusionPipeline, ModelMixin)):
        _export_diffusers_checkpoint(model, dtype, export_dir, components)
        return

    # Transformers model export
    # NOTE: (hg) Early exit for speculative decoding models
    # This is a temp workaround to avoid error with offline spec ckpt during export
    if spec_opt_only(model):
        save_file(export_spec_ckpt_state_dict(model), f"{export_dir}/model.safetensors")
        with open(f"{export_dir}/config.json", "w") as file:
            json.dump(export_spec_ckpt_config(model), file, indent=4)
        return

    try:
        post_state_dict, hf_quant_config = _export_transformers_checkpoint(model, dtype)

        if hf_quant_config is not None:
            # Save hf_quant_config.json for backward compatibility
            with open(f"{export_dir}/hf_quant_config.json", "w") as file:
                json.dump(hf_quant_config, file, indent=4)

            hf_quant_config = convert_hf_quant_config_format(hf_quant_config)

        # Save model
        model.save_pretrained(
            export_dir, state_dict=post_state_dict, save_modelopt_state=save_modelopt_state
        )

        original_config = f"{export_dir}/config.json"
        config_data = {}

        with open(original_config) as file:
            config_data = json.load(file)

        if hf_quant_config is not None:
            config_data["quantization_config"] = hf_quant_config

        with open(original_config, "w") as file:
            json.dump(config_data, file, indent=4)

    except Exception as e:
        warnings.warn(
            "Cannot export model to the model_config. The modelopt-optimized model state_dict"
            " can be saved with torch.save for further inspection."
        )
        raise e
