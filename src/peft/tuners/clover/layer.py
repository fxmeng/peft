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

import warnings
from typing import Any, Optional
import torch
import torch.nn as nn
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

class CloverLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("clover_R",)

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.num_head = {}
        self.head_dim = {}
        self.head_in = {}
        self.clover_R = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            warnings.warn(
                f"Unsupported layer type '{type(base_layer)}' encountered, proceed at your own risk.", UserWarning
            )

        self.in_features = in_features
        self.out_features = out_features
        
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

class Linear(nn.Module, CloverLayer):
    def __init__(
        self,
        base_layer,
        adapter_name: str = 'default',
        head_dim: int = 128,
        head_in: bool = False,
        init_clover_weights: str = 'eye', # Choices: ['eye','qr','absorb-decompose']
        **kwargs,
    ) -> None:
        super().__init__()
        CloverLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            head_dim,
            head_in,
            init_clover_weights=init_clover_weights,
        )

    def update_layer(
        self, adapter_name, head_dim, head_in, init_clover_weights
    ):
        self.head_dim[adapter_name] = head_dim
        self.head_in[adapter_name] = head_in
        # Actual trainable parameters
        if head_in:
            assert self.in_features % head_dim == 0
            self.num_head[adapter_name] = self.in_features // head_dim
        else:
            assert self.out_features % head_dim == 0
            self.num_head[adapter_name] = self.out_features // head_dim
        weight_R = torch.randn((self.num_head[adapter_name], head_dim, head_dim))
        self.clover_R[adapter_name] = nn.Parameter(weight_R)

        # for inits that require access to the base weight, use gather_param_ctx so that the weight is gathered when using DeepSpeed
        if init_clover_weights == "qr":
            self.qr_decompose_init(adapter_name)
        elif init_clover_weights == "absorb-decompose":
            self.absorb_decompose_init(adapter_name, init_clover_weights)
        else:
            self.reset_clover_parameters(adapter_name)

        self.set_adapter(self.active_adapters)

    def reset_clover_parameters(self, adapter_name):
        if adapter_name in self.clover_R.keys():
            weight_R = torch.eye(self.head_dim[adapter_name]).unsqueeze(0).repeat(self.num_head[adapter_name], 1, 1)
            self.clover_R[adapter_name].data = weight_R

    def qr_decompose_init(self, adapter_name):
        dtype = self.base_layer.weight.dtype
        base_weight = self.base_layer.weight.data # (out_dim, in_dim)
        if self.head_in[adapter_name]:
            base_weight = base_weight.view(-1, self.num_head[adapter_name], self.head_dim[adapter_name]).transpose(0,1) # (num_heads, out_dim, head_dim)
            Q, R = torch.linalg.qr(base_weight.to(torch.float32)) # Q(num_heads, out_dim, head_dim), R(num_heads, head_dim, head_dim)
            self.clover_R[adapter_name].data = R.transpose(1,2).to(dtype).contiguous()
            self.base_layer.weight.data = Q.transpose(0,1).reshape(-1, self.num_head[adapter_name]*self.head_dim[adapter_name]).to(dtype).contiguous()
        else:        
            if self.base_layer.bias is not None:
                base_bias = self.base_layer.bias.data.unsqueeze(1) # (out_dim, 1)
                base_weight = torch.cat([base_weight, base_bias],dim=1)  # (out_dim, in_dim+1)
                
            base_weight = base_weight.view(self.num_head[adapter_name], self.head_dim[adapter_name], -1).transpose(1,2)  # (num_heads, in_dim, head_dim) or (num_heads, in_dim+1, head_dim)
            Q, R = torch.linalg.qr(base_weight.to(torch.float32)) # Q(num_heads, in_dim, head_dim), R(num_heads, head_dim, head_dim)
            self.clover_R[adapter_name].data = R.to(dtype).contiguous()
            self.base_layer.weight.data = Q.transpose(1,2).reshape(self.num_head[adapter_name]*self.head_dim[adapter_name], -1).to(dtype).contiguous()

    def absorb_decompose_init(adapter_name):
        pass
    
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
            if active_adapter in self.clover_R.keys():
                base_layer = self.get_base_layer()
                # Note that safe_merge will be slower than the normal merge
                # because of the copy operation.
                base_weights = base_layer.weight.data.clone() # (out_dim, in_dim)
                weight_R = self.clover_R[active_adapter].data #(num_head, head_dim, head_dim)
                if self.head_in[active_adapter]:
                    base_weights = base_weights.view(self.out_features, self.num_head[active_adapter], self.head_dim[active_adapter])
                    base_weights = torch.einsum("ohd,hed->ohe", base_weights, weight_R) 
                else:
                    if base_layer.bias is not None:
                        base_bias = base_layer.bias.data.clone().unsqueeze(1) # (out_dim, 1)
                        base_weights = torch.cat([base_weights, base_bias], dim=1)
                        
                    base_weights = base_weights.view(self.num_head[active_adapter], self.head_dim[active_adapter], -1)
                    base_weights = torch.einsum("hdi,hde->hei", base_weights, weight_R)
                base_weights = base_weights.reshape(self.out_features, -1).contiguous()
                if not torch.isfinite(base_weights).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )
                if base_layer.bias is not None and not self.head_in[active_adapter]:
                    base_layer.bias.data = base_weights[:,-1].contiguous()
                    base_weights = base_weights[:,:-1].contiguous()
                base_layer.weight.data = base_weights

                self.merged_adapters.append(active_adapter)

    def rotation(self, result, clover_R, num_head, head_dim):
        bsz, seq, _ = result.shape
        result = result.view(bsz, seq, num_head, head_dim)
        result = torch.einsum("bshd,hde->bshe", result, clover_R)
        result = result.reshape(bsz, seq, num_head*head_dim).contiguous() 
        return result

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            torch_x_dtype = x.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.clover_R.keys():
                    continue
                if not self.head_in[active_adapter]:
                    continue
                clover_R = self.clover_R[active_adapter]
                x = self.rotation(x, clover_R, self.num_head[active_adapter], self.head_dim[active_adapter])
            x = x.to(torch_x_dtype)
                
            result = self.base_layer(x, *args, **kwargs) # (bsz, seq, num_heads*head_dim)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.clover_R.keys():
                    continue
                if self.head_in[active_adapter]:
                    continue
                clover_R = self.clover_R[active_adapter]
                result = self.rotation(result, clover_R, self.num_head[active_adapter], self.head_dim[active_adapter])
            result = result.to(torch_result_dtype)

        return result

    def _mixed_batch_forward(
        self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])
            
        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.clover_R.keys():
                continue
            if not self.head_in[active_adapter]:
                    continue
            clover_R = self.clover_R[active_adapter]
            torch_x_dtype = x.dtype

            # getting the sub-batch, passing it to CLOVER layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = x[sub_batch_indices_list[i]].to(clover_R.dtype)
            clover_x = self.rotation(sub_batch, clover_R, self.num_head[active_adapter], self.head_dim[active_adapter])
            x[sub_batch_indices_list[i]] = clover_x.to(torch_x_dtype)
            
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.clover_R.keys():
                continue
            if self.head_in[active_adapter]:
                    continue
            clover_R = self.clover_R[active_adapter]

            # getting the sub-batch, passing it to CLOVER layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = result[sub_batch_indices_list[i]].to(clover_R.dtype)
            if self.head_in[active_adapter]:
                    continue
            clover_output = self.rotation(sub_batch, clover_R, self.num_head[active_adapter], self.head_dim[active_adapter])
            result[sub_batch_indices_list[i]] = clover_output.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "clover." + rep
    
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
