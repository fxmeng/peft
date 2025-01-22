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
import warnings
from typing import Any, Optional
import torch
import torch.nn as nn
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

class CrossoverLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("crossover_U", "crossover_S", "crossover_V")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.block_size = {}
        self.alpha = {}
        self.scaling = {}
        self.dropout = nn.ModuleDict({})
        self.crossover_U = nn.ParameterDict({})
        self.crossover_S = nn.ParameterDict({})
        self.crossover_V = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
        else:
            warnings.warn(
                f"Unsupported layer type '{type(base_layer)}' encountered, proceed at your own risk.", UserWarning
            )
        
    def _check_forward_args(self, x, *args, **kwargs):
        """Check if the arguments are compatible with the configs and state of the model"""
        adapter_names = kwargs.get("adapter_names", None)
        if adapter_names is None:
            return

        if len(x) != len(adapter_names):
            msg = (
                "Length of `adapter_names` should be the same as the number of inputs, but got "
                f"{len(adapter_names)} and {len(x)} respectively."
            )
            raise ValueError(msg)

        if self.merged:
            # It is unclear what would be the right thing to do if users pass adapter_names and there are merged
            # adapters. Therefore, it is better to raise an error in this case.
            msg = "Cannot pass `adapter_names` when there are merged adapters, please call `unmerge_adapter` first."
            raise ValueError(msg)

class Linear(nn.Module, CrossoverLayer):
    def __init__(
        self,
        base_layer,
        adapter_name: str = 'default',
        block_size: int = 64,
        alpha: int = 64,
        dropout: float = 0.0,
        init_crossover_weights: str = 'kaiming', # Choices: ['kaiming', 'gaussian', 'orthogonal']
        **kwargs,
    ) -> None:
        super().__init__()
        CrossoverLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            block_size,
            alpha,
            dropout,
            init_crossover_weights=init_crossover_weights,
        )

    def update_layer(
        self, adapter_name, block_size, alpha, dropout, init_crossover_weights
    ):
        self.block_size[adapter_name] = block_size
        self.alpha[adapter_name] = alpha
        if dropout > 0.0:
            dropout_layer = nn.Dropout(p=dropout)
        else:
            dropout_layer = nn.Identity()

        self.dropout.update(nn.ModuleDict({adapter_name: dropout_layer}))
        self.scaling[adapter_name] = alpha / block_size
        # Actual trainable parameters
        self.crossover_U[adapter_name] = nn.Parameter(torch.randn(block_size, self.in_features//block_size, self.out_features//block_size))
        self.crossover_S[adapter_name] = nn.Parameter(torch.ones(1))
        self.crossover_V[adapter_name] = nn.Parameter(torch.randn((self.out_features//block_size, block_size, block_size)))

        # for inits that require access to the base weight, use gather_param_ctx so that the weight is gathered when using DeepSpeed
        if init_crossover_weights == 'kaiming':
            self.kaiming_init(adapter_name)
        elif init_crossover_weights == 'gaussian':
            self.gaussian_init(adapter_name)
        elif init_crossover_weights == 'orthogonal':
            self.orthogonal_init(adapter_name)
        else:
            raise ValueError

        self.set_adapter(self.active_adapters)

    def kaiming_init(self, adapter_name):
        nn.init.kaiming_uniform_(self.crossover_V[adapter_name], a=math.sqrt(5))
        nn.init.zeros_(self.crossover_V[adapter_name])
            
    def gaussian_init(self, adapter_name):
        nn.init.normal_(self.crossover_U[adapter_name], std=1 / self.block_size[adapter_name])
        nn.init.zeros_(self.crossover_V[adapter_name])
            
    def orthogonal_init(self, adapter_name):
        Ua,_,Va = torch.linalg.svd(self.crossover_U[adapter_name],full_matrices=False)
        if self.in_features > self.out_features:
            self.crossover_U[adapter_name].data = Ua.contiguous()
        else:
            self.crossover_U[adapter_name].data = Va.contiguous()
        Ub,_,_ = torch.linalg.svd(self.crossover_V[adapter_name],full_matrices=False)
        self.crossover_V[adapter_name].data = Ub.contiguous()
        base_layer = self.get_base_layer()
        self.crossover_S[adapter_name].data[0] = torch.linalg.norm(base_layer.weight.data, ord=2)
        crossover_weight = self.get_delta_weight(adapter_name)
        print(self.crossover_S[adapter_name].data[0].item())
        base_layer.weight.data -= self.scaling[adapter_name] * crossover_weight.to(base_layer.weight.dtype).to(base_layer.weight.device)
        
    def get_delta_weight(self, adapter_name):
        weight_U = self.crossover_U[adapter_name].data
        full_U = torch.zeros(self.in_features, self.out_features, device=weight_U.device)
        full_U = full_U.reshape(self.block_size[adapter_name], self.in_features//self.block_size[adapter_name], self.block_size[adapter_name], self.out_features//self.block_size[adapter_name])
        for i in range(self.block_size[adapter_name]):
            full_U[i,:,i,:]=weight_U[i]
        full_U = full_U.permute(1,0,3,2).reshape(self.in_features, self.out_features)
        weight_V = self.crossover_V[adapter_name].data
        full_V = torch.zeros(self.out_features, self.out_features, device=weight_U.device)
        full_V = full_V.reshape(self.out_features//self.block_size[adapter_name], self.block_size[adapter_name], self.out_features//self.block_size[adapter_name], self.block_size[adapter_name])
        for i in range(self.out_features//self.block_size[adapter_name]):
            full_V[i,:,i,:]=weight_V[i]
        full_V = full_V.reshape(self.out_features, self.out_features)
        crossover_weight = (full_U @ full_V).T * self.scaling[adapter_name] * self.crossover_S[adapter_name].data[0]
        if not torch.isfinite(crossover_weight).all():
            raise ValueError(
                f"NaNs detected in the merged weights. The adapter {adapter_name} seems to be broken"
            )
        return crossover_weight.contiguous()
    
    def merge(self, safe_merge: bool = True, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `True`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.crossover_U.keys():
                base_layer = self.get_base_layer()
                base_layer.weight.data += self.get_delta_weight(active_adapter).to(base_layer.weight.dtype).to(base_layer.weight.device)
                self.merged_adapters.append(active_adapter)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            torch_result_shape = result.shape
            for active_adapter in self.active_adapters:
                crossover_U = self.crossover_U[active_adapter]
                x = x.reshape(torch_result_shape[0], torch_result_shape[1], crossover_U.shape[1], self.in_features//crossover_U.shape[1])
                x = torch.einsum("bnid,dio->bnod", x, crossover_U)
                x = self.dropout[active_adapter](x)
                crossover_V = self.crossover_V[active_adapter]
                x = torch.einsum("bnod,ode->bnoe", x, crossover_V)
                scaling = self.scaling[active_adapter] * self.crossover_S[active_adapter][0]
                result += scaling * x.reshape(torch_result_shape[0], torch_result_shape[1], self.out_features)
            result = result.to(torch_result_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "crossover." + rep
    
def dispatch_default(
    target: torch.nn.Module,
    adapter_name: str,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Linear):
        new_module = Linear(target, adapter_name, **kwargs)
        
    return new_module
