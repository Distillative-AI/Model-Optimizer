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
# mypy: ignore-errors

"""Provides a function to register activation hooks for a model.
Activation hooks are used to compute activation scores for pruning."""

from typing import Type

from modelopt.torch.nas.plugins.megatron_hooks.base_hooks import ForwardHook as ActivationsHook
from modelopt.torch.puzzletron.tools.logger import aprint


def register_activation_hooks(
    model,
    activation_hooks_kwargs: dict,
    pruning_mixin,
    hook_class: Type[ActivationsHook],
) -> dict[str, ActivationsHook]:
    """Register activation hooks using the pruning mixin approach.

    Args:
        model: The model to register hooks on.
        activation_hooks_kwargs: Keyword arguments passed to hook constructors.
        pruning_mixin: The pruning mixin that defines which modules to hook.
        hook_class: The hook class to instantiate for each module.

    Returns:
        Dictionary mapping module names to hook instances.
    """
    activation_hooks_kwargs["model"] = model

    if hook_class not in pruning_mixin.supported_hooks():
        raise ValueError(
            f"Hook class not supported for {pruning_mixin.__class__.__name__}, "
            f"must be in {pruning_mixin.supported_hooks()}"
        )

    module_names_to_hook = pruning_mixin.get_module_names_to_hook(model)
    activation_hooks = dict()
    for block_idx, module_name in module_names_to_hook:
        block_config = None
        if block_idx is not None:
            block_config = model.config.block_configs[block_idx]
        curr_activation_hooks_kwargs = {
            **activation_hooks_kwargs,
            "block_config": block_config,
        }

        module = model.get_submodule(module_name)
        hook = hook_class(module, curr_activation_hooks_kwargs)
        module.register_forward_hook(hook)
        activation_hooks[module_name] = hook

    if len(activation_hooks) == 0:
        raise ValueError("couldn't find any hooks")

    aprint(f"Found the following hooks: {activation_hooks.keys()}")
    return activation_hooks
