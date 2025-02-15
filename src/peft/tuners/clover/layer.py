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
    adapter_layer_names = ("clover_S", "clover_V")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.num_head = {}
        self.head_dim = {}
        self.decomp = {}
        self.cross_head = {}
        self.clover_S = nn.ParameterDict({})
        self.clover_V = nn.ParameterDict({})
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
        num_head: int = 32,
        decomp: bool = False,
        cross_head: bool = False,
        init_clover_weights: str = 'eye', # Choices: ['eye','qr','absorb-decompose']
        use_s: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        CloverLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            head_dim,
            num_head,
            decomp,
            cross_head,
            init_clover_weights=init_clover_weights,
            use_s=use_s,
        )

    def update_layer(
        self, adapter_name, head_dim, num_head, decomp, cross_head, init_clover_weights, use_s
    ):
        self.head_dim[adapter_name] = head_dim
        self.num_head[adapter_name] = num_head
        self.decomp[adapter_name] = decomp
        self.cross_head[adapter_name] = cross_head
        # Actual trainable parameters
        if use_s:
            weight_S = torch.randn((self.in_features if decomp else self.out_features))
            self.clover_S[adapter_name] = nn.Parameter(weight_S)
        if cross_head:
            weight_V = torch.randn((self.in_features//num_head if decomp else self.out_features//num_head, num_head, num_head))
        else:
            weight_V = torch.randn((self.in_features//head_dim if decomp else self.out_features//head_dim, head_dim, head_dim))
        self.clover_V[adapter_name] = nn.Parameter(weight_V)

        # for inits that require access to the base weight, use gather_param_ctx so that the weight is gathered when using DeepSpeed
        if init_clover_weights == 'svd':
            self.orthonormal_decompose_init(adapter_name, use_s)
        else:
            self.reset_clover_parameters(adapter_name, use_s)

        self.set_adapter(self.active_adapters)

    def reset_clover_parameters(self, adapter_name, use_s):
        if adapter_name in self.clover_V.keys():
            if use_s:
                torch.nn.init.ones_(self.clover_S[adapter_name].data)
            n,d,_ = self.clover_V[adapter_name].data.shape
            weight_V = torch.eye(d).unsqueeze(0).repeat(n, 1, 1)
            self.clover_V[adapter_name].data = weight_V
 
    def orthonormal_decompose_init(self, adapter_name, use_s):
        dtype = self.base_layer.weight.dtype
        base_weight = self.base_layer.weight.data # (out_dim, in_dim)
        if self.decomp[adapter_name]:
            if self.cross_head[adapter_name]:
                base_weight = base_weight.view(self.out_features, self.num_head[adapter_name], -1) # (out_dim, num_heads, head_dim)
                base_weight = base_weight.permute(2,0,1) # (head_dim, out_dim, num_heads)
                U,S,V = torch.linalg.svd(base_weight.to(torch.float32), full_matrices=False) # U(head_dim, out_dim, num_heads), S(head_dim, num_heads), V(head_dim, num_heads, num_heads)
                if not use_s:
                    V = torch.einsum("dh,dhi->dhi", S, V) 
                U = U.permute(1,2,0) # (out_dim, num_heads, head_dim)
                S = S.T # S(num_heads, head_dim)
            else:
                base_weight = base_weight.view(self.out_features, -1, self.head_dim[adapter_name]) # (out_dim, num_heads, head_dim)
                base_weight = base_weight.transpose(0,1) # (num_heads, out_dim, head_dim)
                U,S,V = torch.linalg.svd(base_weight.to(torch.float32), full_matrices=False) # U(num_heads, out_dim, head_dim), S(num_heads, head_dim), V(num_heads, head_dim, head_dim)
                if not use_s:
                    V = torch.einsum("dh,dhi->dhi", S, V) 
                U = U.transpose(0,1) # (out_dim, num_heads, head_dim)
            self.base_layer.weight.data = U.reshape(self.out_features, -1).to(dtype).contiguous()
            if use_s:
                self.clover_S[adapter_name].data = S.reshape(-1).to(dtype).contiguous()
            self.clover_V[adapter_name].data = V.transpose(1,2).to(dtype).contiguous()
        else:
            if self.base_layer.bias is not None:
                base_bias = self.base_layer.bias.data.unsqueeze(1) # (out_dim, 1)
                base_weight = torch.cat([base_weight, base_bias],dim=1)  # (out_dim, in_dim+1)
                in_features = self.in_features + 1
            else:
                in_features = self.in_features
                
            if self.cross_head[adapter_name]:
                base_weight = base_weight.view(self.num_head[adapter_name], -1, in_features) # (num_heads, head_dim, in_dim)
                base_weight = base_weight.permute(1,2,0)  # (head_dim, in_dim, num_heads) or (head_dim, in_dim+1, num_heads)
                U,S,V = torch.linalg.svd(base_weight.to(torch.float32), full_matrices=False) # U(head_dim, in_dim+1, num_heads), S(head_dim, num_heads), V(head_dim, num_heads, num_heads)
                if not use_s:
                    V = torch.einsum("dh,dhi->dhi", S, V) 
                U = U.permute(2,0,1) # (num_heads, head_dim, in_dim) or (num_heads, head_dim, in_dim+1)
                S = S.T # S(num_heads, head_dim)
            else:
                base_weight = base_weight.view(-1, self.head_dim[adapter_name], in_features) # (num_heads, head_dim, in_dim)
                base_weight = base_weight.transpose(1,2)  # (num_heads, in_dim, head_dim) or (num_heads, in_dim+1, head_dim)
                U,S,V = torch.linalg.svd(base_weight.to(torch.float32), full_matrices=False) # U(num_heads, in_dim+1, head_dim), S(num_heads, head_dim), V(num_heads, head_dim, head_dim)
                if not use_s:
                    V = torch.einsum("dh,dhi->dhi", S, V) 
                U = U.transpose(1,2)# (num_heads, head_dim, in_dim) or (num_heads, head_dim, in_dim+1)
            if use_s:
                self.clover_S[adapter_name].data = S.reshape(-1).to(dtype).contiguous()
            self.clover_V[adapter_name].data = V.to(dtype).contiguous()
            if self.base_layer.bias is not None:
                self.base_layer.bias.data = U[:,:,-1].reshape(-1).to(dtype).contiguous()
                U = U[:,:,:-1] # (num_heads, head_dim, in_dim)
            self.base_layer.weight.data = U.reshape(-1, self.in_features).to(dtype).contiguous()
    
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
            if active_adapter in self.clover_V.keys():
                base_layer = self.get_base_layer()
                # Note that safe_merge will be slower than the normal merge
                # because of the copy operation.
                base_weights = base_layer.weight.data.clone() # (out_dim, in_dim)
                if active_adapter in self.clover_S.keys():
                    weight_S = self.clover_S[active_adapter].data #(num_head*head_dim)
                weight_V = self.clover_V[active_adapter].data #(num_head, head_dim, head_dim) or (head_dim, num_head, num_head)
                if self.decomp[active_adapter]:
                    if self.cross_head[active_adapter]:
                        base_weights = base_weights.view(self.out_features, self.num_head[active_adapter], -1)
                        if active_adapter in self.clover_S.keys():
                            weight_S = weight_S.view(self.num_head[active_adapter], -1)
                            base_weights = torch.einsum("ohd,hd,dlh->old", base_weights, weight_S, weight_V) 
                        else:
                            base_weights = torch.einsum("ohd,dlh->old", base_weights, weight_V) 
                    else:
                        base_weights = base_weights.view(self.out_features, -1, self.head_dim[active_adapter])
                        if active_adapter in self.clover_S.keys():
                            weight_S =  weight_S.view(-1, self.head_dim[active_adapter])
                            base_weights = torch.einsum("ohd,hd,hed->ohe", base_weights, weight_S, weight_V) 
                        else:
                            base_weights = torch.einsum("ohd,hed->ohe", base_weights, weight_V) 
                else:
                    if base_layer.bias is not None:
                        base_bias = base_layer.bias.data.clone().unsqueeze(1) # (out_dim, 1)
                        base_weights = torch.cat([base_weights, base_bias], dim=1)
                        in_features = self.in_features + 1
                    else:
                        in_features = self.in_features
                        
                    if self.cross_head[active_adapter]:
                        base_weights = base_weights.view(self.num_head[active_adapter], -1, in_features)
                        if active_adapter in self.clover_S.keys():
                            weight_S =  weight_S.view(self.num_head[active_adapter], -1)
                            base_weights = torch.einsum("hdi,hd,dhl->ldi", base_weights, weight_S, weight_V) 
                        else:
                            base_weights = torch.einsum("hdi,dhl->ldi", base_weights, weight_V) 
                            
                    else:
                        base_weights = base_weights.view(-1, self.head_dim[active_adapter], in_features)
                        if active_adapter in self.clover_S.keys():
                            weight_S =  weight_S.view(-1, self.head_dim[active_adapter])
                            base_weights = torch.einsum("hdi,hd,hde->hei", base_weights, weight_S, weight_V) 
                        else:
                            base_weights = torch.einsum("hdi,hde->hei", base_weights, weight_V) 
                base_weights = base_weights.reshape(self.out_features, -1).contiguous()
                if not torch.isfinite(base_weights).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )
                if base_layer.bias is not None and not self.decomp[active_adapter]:
                    base_layer.bias.data = base_weights[:,-1].contiguous()
                    base_weights = base_weights[:,:-1].contiguous()
                base_layer.weight.data = base_weights

                self.merged_adapters.append(active_adapter)

    def rotation(self, result, clover_V, num_head, head_dim, cross_head=False):
        bsz, seq, _ = result.shape
        if cross_head:
            result = result.view(bsz, seq, num_head, -1)
            result = torch.einsum("bshd,dhi->bsid", result, clover_V)
        else:
            result = result.view(bsz, seq, -1, head_dim)
            result = torch.einsum("bshd,hde->bshe", result, clover_V)
        result = result.reshape(bsz, seq, -1).contiguous() 
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
                if active_adapter not in self.clover_V.keys():
                    continue
                if not self.decomp[active_adapter]:
                    continue
                clover_V = self.clover_V[active_adapter]
                x = self.rotation(x, clover_V, self.num_head[active_adapter], self.head_dim[active_adapter], self.cross_head[active_adapter])
                if active_adapter in self.clover_S.keys():
                    clover_S = self.clover_S[active_adapter]
                    x *= clover_S
            x = x.to(torch_x_dtype)
            result = self.base_layer(x, *args, **kwargs) # (bsz, seq, num_heads*head_dim)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.clover_V.keys():
                    continue
                if self.decomp[active_adapter]:
                    continue
                clover_V = self.clover_V[active_adapter]
                if active_adapter in self.clover_S.keys():
                    clover_S = self.clover_S[active_adapter]
                    result *= clover_S
                result = self.rotation(result, clover_V, self.num_head[active_adapter], self.head_dim[active_adapter], self.cross_head[active_adapter])
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
            if active_adapter not in self.clover_V.keys():
                continue
            if not self.decomp[active_adapter]:
                    continue
            
            clover_V = self.clover_V[active_adapter]
            torch_x_dtype = x.dtype

            # getting the sub-batch, passing it to CLOVER layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = x[sub_batch_indices_list[i]].to(clover_V.dtype)
            clover_x = self.rotation(sub_batch, clover_V, self.num_head[active_adapter], self.head_dim[active_adapter], self.cross_head[active_adapter])
            if active_adapter in self.clover_S.keys():
                clover_S = self.clover_S[active_adapter]
                clover_x *= clover_S
            x[sub_batch_indices_list[i]] = clover_x.to(torch_x_dtype)
            
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.clover_V.keys():
                continue
            if self.decomp[active_adapter]:
                    continue
            clover_V = self.clover_V[active_adapter]

            # getting the sub-batch, passing it to CLOVER layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = result[sub_batch_indices_list[i]].to(clover_V.dtype)
            if self.decomp[active_adapter]:
                    continue
            if active_adapter in self.clover_S.keys():
                clover_S = self.clover_S[active_adapter]
                sub_batch *= clover_S
            clover_output = self.rotation(sub_batch, clover_V, self.num_head[active_adapter], self.head_dim[active_adapter], self.cross_head[active_adapter])
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
