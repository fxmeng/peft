# Copyright 2023-present the HuggingFace Inc. team.
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
from __future__ import annotations

import math
import operator
import re
import warnings
from contextlib import contextmanager
from dataclasses import asdict, replace
from enum import Enum
from functools import partial, reduce
from itertools import chain
from typing import Literal, Optional

import torch
from torch import nn
from tqdm import tqdm

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
    check_target_module_exists,
    onload_layer,
    replicate_layers,
)
from peft.utils import (
    TRANSFORMERS_MODELS_TO_PiSSA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _freeze_adapter,
    _get_submodules,
    get_peft_model_state_dict,
)
from peft.utils.merge_utils import dare_linear, dare_ties, magnitude_prune, task_arithmetic, ties
from .config import PiSSAConfig
from .layer import Conv2d, PiSSALayer, dispatch_default


def _adapter_names_pre_forward_hook(target, args, kwargs, adapter_names):
    # pre-forward hook to inject the adapter_names argument when using mixed adapter batches inference
    kwargs["adapter_names"] = adapter_names
    return args, kwargs


class PiSSAModel(BaseTuner):
    """
    Creates Principal Singular values and Singular vectors Adaptation (PiSSA) model from a pretrained transformers model.

    The method is described in detail in https://arxiv.org/abs/2404.02948.

    Args:
        model ([`torch.nn.Module`]): The model to be adapted.
        config ([`PiSSAConfig`]): The configuration of the PiSSA model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        `torch.nn.Module`: The PiSSA model.

    Example:

        ```py
        >>> import torch
        >>> import transformers
        >>> from peft import PiSSAConfig, PeftModel, get_peft_model
        >>> target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        >>> config = PiSSAConfig(r=16, target_modules=target_modules, task_type="CAUSAL_LM")
        >>> model = transformers.AutoModelForCausalLM.from_pretrained(
            ...     "meta-llama/Meta-Llama-3-8B",
            ...     device_map="auto",
            ...     torch_dtype=torch.bfloat16,
        >>> )
        >>> pissa_model = get_peft_model(model, config)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`PiSSAConfig`]): The configuration of the PiSSA model.
    """

    prefix: str = "pissa_"

    def __init__(self, model, config, adapter_name) -> None:
        super().__init__(model, config, adapter_name)

    def _check_new_adapter_config(self, config: PiSSAConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        # TODO: there should be a check if any of the existing adapters actually has bias != "none", or else the check
        # does not fully correspond to the error message.
        if (len(self.peft_config) > 1) and (config.bias != "none"):
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. When using multiple adapters, "
                "set bias to 'none' for all adapters."
            )

    @staticmethod
    def _check_target_module_exists(pissa_config, key):
        return check_target_module_exists(pissa_config, key)

    def _prepare_model(self, peft_config: PiSSAConfig, model: nn.Module):
        r"""
        A private method to modify the model structure before adapter is applied.

        Args:
            peft_config (`PeftConfig`):
                The prepared adapter config.
            model (`nn.Module`):
                The model that is going to be adapted.
        """
        if peft_config.layer_replication:
            replicate_layers(model, peft_config.layer_replication)

    def _create_and_replace(
        self,
        pissa_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # Regexp matching - Find key which matches current target_name in patterns provided
        pattern_keys = list(chain(pissa_config.rank_pattern.keys(), pissa_config.alpha_pattern.keys()))
        target_name_key = next(filter(lambda key: re.match(rf".*\.{key}$", current_key), pattern_keys), current_key)
        r = pissa_config.rank_pattern.get(target_name_key, pissa_config.r)

        kwargs = {
            "r": r,
            "pissa_alpha": pissa_config.pissa_alpha,
            "pissa_dropout": pissa_config.pissa_dropout,
            "singular_value": pissa_config.singular_value,
            "fan_in_fan_out": pissa_config.fan_in_fan_out,
            "init_pissa_weights": pissa_config.init_pissa_weights,
            "fsvd": pissa_config.fsvd,
            "use_rspissa": pissa_config.use_rspissa,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }

        if isinstance(target, PiSSALayer):
            target.update_layer(
                adapter_name,
                r,
                pissa_alpha=pissa_config.pissa_alpha,
                pissa_dropout=pissa_config.pissa_dropout,
                singular_value=pissa_config.singular_value,
                init_pissa_weights=pissa_config.init_pissa_weights,
                fsvd = pissa_config.fsvd,
                use_rspissa=pissa_config.use_rspissa,
                update_layer=pissa_config.update_layer,
            )
        else:
            new_module = self._create_new_module(pissa_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        if not hasattr(new_module, "base_layer"):
            if hasattr(new_module, "W_q"):  # HQQ
                new_module.W_q = child.W_q
            else:
                new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if (self.prefix in name) or ("ranknum" in name):
                weight = (
                    child.qweight
                    if hasattr(child, "qweight")
                    else child.W_q
                    if hasattr(child, "W_q")
                    else child.weight
                )
                module.to(weight.device)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias == "none":
                continue

            if bias == "all":
                for n, p in model.named_parameters():
                    if "bias" in n:
                        p.requires_grad = True
            elif bias == "pissa_only":
                for m in model.modules():
                    if isinstance(m, PiSSALayer) and hasattr(m, "bias") and m.bias is not None:
                        m.bias.requires_grad = True
            else:
                raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

    @staticmethod
    def _create_new_module(pissa_config, adapter_name, target, **kwargs):
        # Collect dispatcher functions to decide what backend to use for the replaced PiSSA layer. The order matters,
        # because the first match is always used. Therefore, the default layers should be checked last.
        dispatchers = []

        # avoid eager bnb import
        if is_bnb_available():
            from .bnb import dispatch_bnb_8bit

            dispatchers.append(dispatch_bnb_8bit)

        if is_bnb_4bit_available():
            from .bnb import dispatch_bnb_4bit

            dispatchers.append(dispatch_bnb_4bit)

        dispatchers.extend(
            [
                dispatch_default,
            ]
        )

        new_module = None
        for dispatcher in dispatchers:
            new_module = dispatcher(target, adapter_name, pissa_config=pissa_config, **kwargs)
            if new_module is not None:  # first match wins
                break

        if new_module is None:
            # no module could be matched
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv2d`, `transformers.pytorch_utils.Conv1D`."
            )

        return new_module

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled: bool = True) -> None:
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def enable_adapter_layers(self) -> None:
        """Enable all adapters.

        Call this if you have previously disabled all adapters and want to re-enable them.
        """
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self) -> None:
        """Disable all adapters.

        When disabling all adapters, the model output corresponds to the output of the base model.
        """
        for active_adapter in self.active_adapters:
            val = self.peft_config[active_adapter].bias
            if val != "none":
                msg = (
                    f"Careful, disabling adapter layers with bias configured to be '{val}' does not produce the same "
                    "output as the the base model would without adaption."
                )
                warnings.warn(msg)
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name: str | list[str]) -> None:
        """Set the active adapter(s).

        Additionally, this function will set the specified adapters to trainable (i.e., requires_grad=True). If this is
        not desired, use the following code.

        ```py
        >>> for name, param in model_peft.named_parameters():
        ...     if ...:  # some check on name (ex. if 'pissa' in name)
        ...         param.requires_grad = False
        ```

        Args:
            adapter_name (`str` or `list[str]`): Name of the adapter(s) to be activated.
        """
        for module in self.model.modules():
            if isinstance(module, PiSSALayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    @contextmanager
    def _enable_peft_forward_hooks(self, *args, **kwargs):
        # If adapter_names is passed as an argument, we inject it into the forward arguments.
        adapter_names = kwargs.pop("adapter_names", None)
        if adapter_names is None:
            # nothing to do
            yield
            return

        if self.training:
            raise ValueError("Cannot pass `adapter_names` when the model is in training mode.")

        hook_handles = []
        for module in self.modules():
            if isinstance(module, PiSSALayer):
                pre_forward = partial(_adapter_names_pre_forward_hook, adapter_names=adapter_names)
                handle = module.register_forward_pre_hook(pre_forward, with_kwargs=True)
                hook_handles.append(handle)

        yield

        for handle in hook_handles:
            handle.remove()

    def _check_merge_allowed(self):
        """Verify that the configuration supports merging.

        Currently gptq quantization and replicated layers do not support merging.
        """
        if getattr(self.model, "quantization_method", None) == "gptq":
            raise ValueError("Cannot merge PiSSA layers when the model is gptq quantized")
        if self.peft_config.get("layer_replication"):
            raise ValueError("Cannot merge PiSSA layers when base model layers are replicated")

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_PiSSA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_PiSSA_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config

    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ):
        if merge:
            self._check_merge_allowed()

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            with onload_layer(target):
                if hasattr(target, "base_layer"):
                    if merge:
                        target.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                    self._replace_module(parent, target_name, target.get_base_layer(), target)
                elif isinstance(target, ModulesToSaveWrapper):
                    # save any additional trainable modules part of `modules_to_save`
                    new_module = target.modules_to_save[target.active_adapter]
                    if hasattr(new_module, "base_layer"):
                        # check if the module is itself a tuner layer
                        if merge:
                            new_module.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                        new_module = new_module.get_base_layer()
                    setattr(parent, target_name, new_module)

        return self.model

    def _check_add_weighted_adapter(
        self, adapters: list[str], combination_type: str, svd_rank: int | None
    ) -> tuple[str, int, str]:
        """
        Helper function to check if the arguments to add_weighted_adapter are valid and compatible with the underlying
        model.
        """
        for adapter in adapters:
            if adapter not in list(self.peft_config.keys()):
                raise ValueError(f"Adapter {adapter} does not exist")

        # If more than one of the adapters targets the same module with modules_to_save, raise an error, as these
        # modules cannot be merged. First, find the ModulesToSaveWrapper instances in the model, then check if they
        # have modules for the adapters to be merged.
        modules_to_save_wrappers = [module for module in self.modules() if isinstance(module, ModulesToSaveWrapper)]
        problematic_wrappers = [
            wrapper
            for wrapper in modules_to_save_wrappers
            if sum(adapter in wrapper.modules_to_save for adapter in adapters) > 1
        ]
        if problematic_wrappers:
            raise ValueError(
                "Cannot add weighted adapters if they target the same module with modules_to_save, but found "
                f"{len(problematic_wrappers)} such instance(s)."
            )

        # if there is only one adapter, we can only use linear merging
        combination_type = "linear" if len(adapters) == 1 else combination_type

        adapters_ranks = [self.peft_config[adapter].r for adapter in adapters]
        if combination_type in ("linear", "ties", "dare_ties", "dare_linear", "magnitude_prune"):
            # all adapters ranks should be same, new rank is just this value
            if len(set(adapters_ranks)) != 1:
                raise ValueError(
                    "All adapters must have the same r value when using combination_type linear, ties, dare_ties or "
                    "dare_linear."
                )
            new_rank = adapters_ranks[0]
        elif combination_type == "cat":
            # adapters ranks may be different, new rank is sum of all ranks
            # be careful, because output adapter rank may be really big if mixing a lot of adapters
            new_rank = sum(adapters_ranks)
        elif combination_type.endswith("svd"):
            # new rank is the max of all ranks of the adapters if not provided
            new_rank = svd_rank or max(adapters_ranks)
        else:
            raise ValueError(f"Invalid combination_type: {combination_type}")

        target_module_types = [type(self.peft_config[adapter].target_modules) for adapter in adapters]
        if not target_module_types:
            raise ValueError(f"Found no adapter matching the names in {adapters}")
        if len(set(target_module_types)) > 1:
            raise ValueError(
                "all adapter configs should follow the same target modules type. "
                "Combining adapters with `target_modules` type being a mix of list/set and string is not supported."
            )

        if target_module_types[0] == str:
            new_target_modules = "|".join(f"({self.peft_config[adapter].target_modules})" for adapter in adapters)
        elif target_module_types[0] == set:
            new_target_modules = reduce(
                operator.or_, (self.peft_config[adapter].target_modules for adapter in adapters)
            )
        else:
            raise TypeError(f"Invalid type {target_module_types[0]} found in target_modules")

        return combination_type, new_rank, new_target_modules

    def add_weighted_adapter(
        self,
        adapters: list[str],
        weights: list[float],
        adapter_name: str,
        combination_type: str = "svd",
        svd_rank: int | None = None,
        svd_clamp: int | None = None,
        svd_full_matrices: bool = True,
        svd_driver: str | None = None,
        density: float | None = None,
        majority_sign_method: Literal["total", "frequency"] = "total",
    ) -> None:
        """
        This method adds a new adapter by merging the given adapters with the given weights.

        When using the `cat` combination_type you should be aware that rank of the resulting adapter will be equal to
        the sum of all adapters ranks. So it's possible that the mixed adapter may become too big and result in OOM
        errors.

        Args:
            adapters (`list`):
                List of adapter names to be merged.
            weights (`list`):
                List of weights for each adapter.
            adapter_name (`str`):
                Name of the new adapter.
            combination_type (`str`):
                The merging type can be one of [`svd`, `linear`, `cat`, `ties`, `ties_svd`, `dare_ties`, `dare_linear`,
                `dare_ties_svd`, `dare_linear_svd`, `magnitude_prune`, `magnitude_prune_svd`]. When using the `cat`
                combination_type, the rank of the resulting adapter is equal to the sum of all adapters ranks (the
                mixed adapter may be too big and result in OOM errors).
            svd_rank (`int`, *optional*):
                Rank of output adapter for svd. If None provided, will use max rank of merging adapters.
            svd_clamp (`float`, *optional*):
                A quantile threshold for clamping SVD decomposition output. If None is provided, do not perform
                clamping. Defaults to None.
            svd_full_matrices (`bool`, *optional*):
                Controls whether to compute the full or reduced SVD, and consequently, the shape of the returned
                tensors U and Vh. Defaults to True.
            svd_driver (`str`, *optional*):
                Name of the cuSOLVER method to be used. This keyword argument only works when merging on CUDA. Can be
                one of [None, `gesvd`, `gesvdj`, `gesvda`]. For more info please refer to `torch.linalg.svd`
                documentation. Defaults to None.
            density (`float`, *optional*):
                Value between 0 and 1. 0 means all values are pruned and 1 means no values are pruned. Should be used
                with [`ties`, `ties_svd`, `dare_ties`, `dare_linear`, `dare_ties_svd`, `dare_linear_svd`,
                `magnintude_prune`, `magnitude_prune_svd`]
            majority_sign_method (`str`):
                The method, should be one of ["total", "frequency"], to use to get the magnitude of the sign values.
                Should be used with [`ties`, `ties_svd`, `dare_ties`, `dare_ties_svd`]
        """

        if adapter_name in list(self.peft_config.keys()):
            return

        combination_type, new_rank, new_target_modules = self._check_add_weighted_adapter(
            adapters=adapters,
            combination_type=combination_type,
            svd_rank=svd_rank,
        )

        self.peft_config[adapter_name] = replace(
            self.peft_config[adapters[0]],
            r=new_rank,
            target_modules=new_target_modules,
        )
        self.inject_adapter(self.model, adapter_name)

        # Do we really need that?
        _freeze_adapter(self.model, adapter_name)

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, PiSSALayer):
                if adapter_name in target.pissa_A:
                    target_pissa_A = target.pissa_A[adapter_name].weight
                    target_pissa_B = target.pissa_B[adapter_name].weight
                elif adapter_name in target.pissa_embedding_A:
                    target_pissa_A = target.pissa_embedding_A[adapter_name]
                    target_pissa_B = target.pissa_embedding_B[adapter_name]
                else:
                    continue

                target_pissa_A.data = target_pissa_A.data * 0.0
                target_pissa_B.data = target_pissa_B.data * 0.0
                if combination_type == "cat":
                    pissas_A, pissas_B = [], []
                    for adapter, weight in zip(adapters, weights):
                        if adapter in target.pissa_A:
                            current_adapter_pissa_A = target.pissa_A[adapter].weight
                            current_adapter_pissa_B = target.pissa_B[adapter].weight
                        elif adapter in target.pissa_embedding_A:
                            current_adapter_pissa_A = target.pissa_embedding_A[adapter]
                            current_adapter_pissa_B = target.pissa_embedding_B[adapter]
                        else:
                            continue
                        pissas_A.append(current_adapter_pissa_A.data * weight * target.scaling[adapter])
                        pissas_B.append(current_adapter_pissa_B.data)

                    if len(pissas_A) == 0:
                        raise ValueError("No matching PiSSAs found. Please raise an issue on GitHub.")
                    pissas_A = torch.cat(pissas_A, dim=0)
                    pissas_B = torch.cat(pissas_B, dim=1)
                    target_pissa_A.data[: pissas_A.shape[0], :] = pissas_A
                    target_pissa_B.data[:, : pissas_B.shape[1]] = pissas_B
                elif combination_type in [
                    "svd",
                    "ties_svd",
                    "dare_linear_svd",
                    "dare_ties_svd",
                    "magnitude_prune_svd",
                ]:
                    target_pissa_A.data, target_pissa_B.data = self._svd_generalized_task_arithmetic_weighted_adapter(
                        combination_type,
                        adapters,
                        weights,
                        new_rank,
                        target,
                        target_pissa_A,
                        target_pissa_B,
                        density,
                        majority_sign_method,
                        svd_clamp,
                        full_matrices=svd_full_matrices,
                        driver=svd_driver,
                    )
                elif combination_type in ["linear", "ties", "dare_linear", "dare_ties", "magnitude_prune"]:
                    target_pissa_A.data, target_pissa_B.data = self._generalized_task_arithmetic_weighted_adapter(
                        combination_type, adapters, weights, target, density, majority_sign_method
                    )

    def _svd_generalized_task_arithmetic_weighted_adapter(
        self,
        combination_type,
        adapters,
        weights,
        new_rank,
        target,
        target_pissa_A,
        target_pissa_B,
        density,
        majority_sign_method,
        clamp=None,
        full_matrices=True,
        driver=None,
    ):
        valid_adapters = []
        valid_weights = []
        is_embedding = any(adapter in target.pissa_embedding_A for adapter in adapters)
        for adapter, weight in zip(adapters, weights):
            if adapter in target.pissa_A or adapter in target.pissa_embedding_A:
                valid_adapters.append(adapter)
                valid_weights.append(weight * target.scaling[adapter])

        # if no valid adapter, nothing to do
        if len(valid_adapters) == 0:
            raise ValueError("No matching PiSSAs found. Please raise an issue on Github.")
        delta_weight = [target.get_delta_weight(adapter) for adapter in valid_adapters]
        valid_weights = torch.tensor(valid_weights).to(delta_weight[0].device)
        if combination_type == "svd":
            delta_weight = task_arithmetic(delta_weight, valid_weights)
        elif combination_type == "ties_svd":
            delta_weight = ties(delta_weight, valid_weights, density, majority_sign_method)
        elif combination_type == "dare_linear_svd":
            delta_weight = dare_linear(delta_weight, valid_weights, density)
        elif combination_type == "dare_ties_svd":
            delta_weight = dare_ties(delta_weight, valid_weights, density, majority_sign_method)
        elif combination_type == "magnitude_prune_svd":
            delta_weight = magnitude_prune(delta_weight, valid_weights, density)
        else:
            raise ValueError(f"Invalid value passed to combination type: {combination_type}")

        conv2d = isinstance(target, Conv2d)
        if conv2d:
            conv2d_1x1 = target.weight.size()[2:4] == (1, 1)
            if not conv2d_1x1:
                delta_weight = delta_weight.flatten(start_dim=1)
            else:
                delta_weight = delta_weight.squeeze()
        if (hasattr(target, "fan_in_fan_out") and target.fan_in_fan_out) or is_embedding:
            delta_weight = delta_weight.T

        # based on https://github.com/kohya-ss/sd-scripts/blob/main/networks/svd_merge_pissa.py#L114-L131
        U, S, Vh = torch.linalg.svd(delta_weight, full_matrices=full_matrices, driver=driver)
        U = U[:, :new_rank]
        S = S[:new_rank]
        U = U @ torch.diag(S)
        Vh = Vh[:new_rank, :]
        if clamp is not None:
            dist = torch.cat([U.flatten(), Vh.flatten()])
            hi_val = torch.quantile(dist, clamp)
            low_val = -hi_val
            U = U.clamp(low_val, hi_val)
            Vh = Vh.clamp(low_val, hi_val)
        if conv2d:
            U = U.reshape(target_pissa_B.data.shape)
            Vh = Vh.reshape(target_pissa_A.data.shape)
        return Vh, U

    def _generalized_task_arithmetic_weighted_adapter(
        self,
        combination_type,
        adapters,
        weights,
        target,
        density,
        majority_sign_method,
    ):
        # account weights for PiSSA A and B layers.
        valid_weights = []
        pissa_A_deltas = []
        pissa_B_deltas = []
        for adapter, weight in zip(adapters, weights):
            if adapter in target.pissa_A:
                current_adapter_pissa_A = target.pissa_A[adapter].weight
                current_adapter_pissa_B = target.pissa_B[adapter].weight
            elif adapter in target.pissa_embedding_A:
                current_adapter_pissa_A = target.pissa_embedding_A[adapter]
                current_adapter_pissa_B = target.pissa_embedding_B[adapter]
            else:
                continue
            valid_weights.append(math.sqrt(weight * target.scaling[adapter]))
            pissa_A_deltas.append(current_adapter_pissa_A.data)
            pissa_B_deltas.append(current_adapter_pissa_B.data)
        valid_weights = torch.tensor(valid_weights).to(pissa_A_deltas[0].device)
        pissa_deltas = [pissa_A_deltas, pissa_B_deltas]
        dtype = pissa_A_deltas[0].dtype
        for i, task_tensors in enumerate(pissa_deltas):
            if combination_type == "linear":
                pissa_deltas[i] = task_arithmetic(task_tensors, valid_weights)
            elif combination_type == "ties":
                pissa_deltas[i] = ties(task_tensors, valid_weights, density, majority_sign_method)
            elif combination_type == "dare_linear":
                pissa_deltas[i] = dare_linear(task_tensors, valid_weights, density)
            elif combination_type == "dare_ties":
                pissa_deltas[i] = dare_ties(task_tensors, valid_weights, density, majority_sign_method)
            elif combination_type == "magnitude_prune":
                pissa_deltas[i] = magnitude_prune(task_tensors, valid_weights, density)
            else:
                raise ValueError("Invalid combination type")
        pissa_deltas = [delta.to(dtype) for delta in pissa_deltas]
        return pissa_deltas

    def delete_adapter(self, adapter_name: str) -> None:
        """
        Deletes an existing adapter.

        Args:
            adapter_name (str): Name of the adapter to be deleted.
        """
        if adapter_name not in list(self.peft_config.keys()):
            raise ValueError(f"Adapter {adapter_name} does not exist")
        del self.peft_config[adapter_name]

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        new_adapter = None
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, PiSSALayer):
                target.delete_adapter(adapter_name)
                if new_adapter is None:
                    new_adapter = target.active_adapters[:]

        self.active_adapter = new_adapter or []

    def merge_and_unload(
        self, progressbar: bool = False, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
    ) -> torch.nn.Module:
        r"""
        This method merges the PiSSA layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.

        Args:
            progressbar (`bool`):
                whether to show a progressbar indicating the unload and merge process
            safe_merge (`bool`):
                whether to activate the safe merging check to check if there is any potential Nan in the adapter
                weights
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import PeftModel

        >>> base_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b")
        >>> peft_model_id = "smangrul/falcon-40B-int4-peft-pissa-sfttrainer-sample"
        >>> model = PeftModel.from_pretrained(base_model, peft_model_id)
        >>> merged_model = model.merge_and_unload()
        ```
        """
        return self._unload_and_optionally_merge(
            progressbar=progressbar, safe_merge=safe_merge, adapter_names=adapter_names
        )

    def unload(self) -> torch.nn.Module:
        """
        Gets back the base model by removing all the pissa modules without merging. This gives back the original base
        model.
        """
        return self._unload_and_optionally_merge(merge=False)

    def subtract_mutated_init(self, output_state_dict: dict[str, torch.Tensor], adapter_name: str, kwargs=None):
        """
        This function can calculate the updates of the PiSSA by comparing the parameters of the PiSSA adapter in `output_state_dict` with the initial values of PiSSA in `adapter_name`, thus
        converting PiSSA to  LoRA.
        """
        for name, param in self.model.named_parameters():
            if (
                param.data.dtype != torch.float32
                and param.data.dtype != torch.float16
                and param.data.dtype != torch.bfloat16
            ) and adapter_name.startswith("pissa"):
                warnings.warn(
                    r"Note that Quant(W_res) + USV != Quant(W) + \Delta(USV); "
                    "the converted PiSSA, when combined with W or Quant(W), may introduce a certain gap in the fine-tuned model. "
                    "Therefore, we recommend directly using the Quant(W_res) in conjunction with the PiSSA adapter. "
                )
        mutated_init_state_dict = get_peft_model_state_dict(
            self,
            state_dict=kwargs.get("state_dict", None),
            adapter_name=adapter_name,
        )
        tensors_pissa = {}
        for name in output_state_dict.keys():
            ## W = W^{res} + U_0 \times (S_0 * V_0),
            ## W + \Delta W = W^{res} + U \times (S * V),
            ## \Delta W = U \times (S * V) - U_0 \times (S_0 * V_0) = [U | U_0 ] \times [S * V | -S_0 * V_0]^T = AB.
            if "pissa_U" in name:
                tensors_pissa[name.replace("pissa_U", "lora_A")] = torch.cat(
                    [output_state_dict[name],  mutated_init_state_dict[".".join(name.split(".")[1:])]], dim=0
                )
            elif "pissa_V" in name:
                sname = name.replace("pissa_V.weight", "pissa_S")
                tensors_pissa[name.replace("pissa_V", "lora_B")] = torch.cat(
                    [output_state_dict[name] * output_state_dict[sname], -mutated_init_state_dict[".".join(name.split(".")[1:])] * mutated_init_state_dict[".".join(sname.split(".")[1:])]], dim=1
                )

        return tensors_pissa
