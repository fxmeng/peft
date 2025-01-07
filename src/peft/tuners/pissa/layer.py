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
from typing import Any, Optional, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import svd_lowrank

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose

from .config import PiSSAConfig

class PiSSALayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("pissa_U", "pissa_S", "pissa_V", "pissa_embedding_U", "pissa_embedding_S", "pissa_embedding_V")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "pissa_alpha", "pissa_dropout", "fsvd", "singular_value")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.pissa_alpha = {}
        self.scaling = {}
        self.fsvd = {}
        self.pissa_dropout = nn.ModuleDict({})
        self.pissa_U = nn.ModuleDict({})
        self.pissa_S = nn.ParameterDict({})
        self.pissa_V = nn.ModuleDict({})
        # For Embedding layer
        self.pissa_embedding_U = nn.ParameterDict({})
        self.pissa_embedding_S = nn.ParameterDict({})
        self.pissa_embedding_V = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self._caches: dict[str, Any] = {}
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features

    def _cache_store(self, key: str, value: Any) -> None:
        self._caches[key] = value

    def _cache_pop(self, key: str) -> Any:
        value = self._caches.pop(key)
        return value

    def svd_or_fsvd(self, adapter_name, weight):
        if self.fsvd[adapter_name] is None:
            U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
            Ur = U[:, : self.r[adapter_name]]
            Sr = S[: self.r[adapter_name]]
            Vhr = Vh[: self.r[adapter_name]]
        elif isinstance(self.fsvd[adapter_name], int) and self.fsvd[adapter_name] > 0:
            Ur, Sr, Vr = svd_lowrank(
                weight, self.r[adapter_name], niter = self.fsvd[adapter_name]
            )
            Vhr = Vr.t()
        else:
            raise ValueError(
                f"fsvd should be 'int' or 'None', got {self.fsvd[adapter_name]} instead."
            )
        return Ur.contiguous(), Sr.contiguous(), Vhr.contiguous()

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

        unique_adapters = set(self.active_adapters)

class Linear(nn.Module, PiSSALayer):
    # PiSSA implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        pissa_alpha: int = 8,
        pissa_dropout: float = 0.0,
        singular_value: str = None,
        init_pissa_weights: Union[bool, str] = True,
        fsvd: int = None,
        use_rspissa: bool = False,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        PiSSALayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            pissa_alpha,
            pissa_dropout,
            singular_value,
            init_pissa_weights,
            fsvd=fsvd,
            use_rspissa=use_rspissa,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def update_layer(
        self, adapter_name, r, pissa_alpha, pissa_dropout, singular_value, init_pissa_weights, fsvd, use_rspissa
    ):
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.pissa_alpha[adapter_name] = pissa_alpha
        self.fsvd[adapter_name] = fsvd
        if pissa_dropout > 0.0:
            pissa_dropout_layer = nn.Dropout(p=pissa_dropout)
        else:
            pissa_dropout_layer = nn.Identity()
            
        self.pissa_dropout.update(nn.ModuleDict({adapter_name: pissa_dropout_layer}))
        # Actual trainable parameters
        self.pissa_U[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        if singular_value == "vector":
            self.pissa_S[adapter_name] = nn.Parameter(torch.zeros(r))
        elif singular_value == "matrix":
            self.pissa_S[adapter_name] = nn.Parameter(torch.zeros(r, r))
        self.pissa_V[adapter_name] = nn.Linear(r, self.out_features, bias=False)
        if use_rspissa:
            self.scaling[adapter_name] = pissa_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = pissa_alpha / r

        if init_pissa_weights:
            self.reset_pissa_parameters(adapter_name, singular_value)
        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        self.set_adapter(self.active_adapters)
            
    def reset_pissa_parameters(self, adapter_name, singular_value):
        weight = self.get_base_layer().weight.data
        dtype = weight.dtype
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            raise TypeError(
                "Please initialize PiSSA under float32, float16, or bfloat16. "
                "Subsequently, re-quantize the residual model to help minimize quantization errors."
            )
        weight = weight.to(torch.float32)
        pissa_V, pissa_S, pissa_U = self.svd_or_fsvd(adapter_name, weight)
        if singular_value is None:
            self.pissa_U[adapter_name].weight.data = torch.sqrt(pissa_S).unsqueeze(1) *  pissa_U
        else:
            self.pissa_U[adapter_name].weight.data = pissa_U
            
        if singular_value == "vector":
            self.pissa_S[adapter_name].data = pissa_S
        elif singular_value == "matrix":
            self.pissa_S[adapter_name].data = torch.diag(pissa_S)
        
        if singular_value is None:
            self.pissa_V[adapter_name].weight.data = pissa_V * torch.sqrt(pissa_S)
        else:
            self.pissa_V[adapter_name].weight.data = pissa_V
        
        weight = weight.data - pissa_V * pissa_S @ pissa_U * self.scaling[adapter_name]
        weight = weight.to(dtype)
        self.get_base_layer().weight.data = weight

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.pissa_U.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    orig_weights = orig_weights + delta_weight

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    base_layer.weight.data = base_layer.weight.data + delta_weight

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.pissa_U.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                weight.data -= delta_weight

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.pissa_V[adapter].weight.device
        dtype = self.pissa_V[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_U = self.pissa_U[adapter].weight
        if adapter in self.pissa_S:
            weight_S = self.pissa_S[adapter]
        weight_V = self.pissa_V[adapter].weight

        if cast_to_fp32:
            weight_U = weight_U.float()
            if adapter in self.pissa_S:
                weight_S = weight_S.float()
            weight_V = weight_V.float()

        if adapter in self.pissa_S:
            if len(weight_S.shape)==2:
                before_V = weight_S @ weight_U
            else:
                before_V = weight_S.unsqueeze(1) *  weight_U
        else:
            before_V = weight_U
        output_tensor = transpose(weight_V @ before_V, self.fan_in_fan_out)
            

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.pissa_U[adapter].weight.data = weight_U.to(dtype)
            if adapter in self.pissa_S:
                self.pissa_S[adapter].data = weight_S.to(dtype)
            self.pissa_V[adapter].weight.data = weight_V.to(dtype)

        return output_tensor

    def _mixed_batch_forward(
        self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.pissa_U.keys():
                continue
            pissa_dropout = self.pissa_dropout[active_adapter]
            pissa_U = self.pissa_U[active_adapter]
            pissa_V = self.pissa_V[active_adapter]
            scaling = self.scaling[active_adapter]
            # getting the sub-batch, passing it to PiSSA layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = pissa_dropout(x[sub_batch_indices_list[i]].to(pissa_U.weight.dtype))
            after_U = pissa_U(sub_batch)
            if active_adapter in self.pissa_S.keys():
                pissa_S = self.pissa_S[active_adapter]
                if len(pissa_S.shape)==2:
                    before_V = after_U @ pissa_S
                else:
                    before_V = after_U * pissa_S
            else:
                before_V = after_U
                
            pissa_output = pissa_V(before_V) * scaling
                
            result[sub_batch_indices_list[i]] += pissa_output.to(torch_result_dtype)

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
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.pissa_U.keys():
                    continue
                pissa_dropout = self.pissa_dropout[active_adapter]
                pissa_U = self.pissa_U[active_adapter]
                pissa_V = self.pissa_V[active_adapter]
                scaling = self.scaling[active_adapter]
                x = pissa_dropout(x.to(pissa_U.weight.dtype))
                after_U = pissa_U(x)
                if active_adapter in self.pissa_S.keys():
                    pissa_S = self.pissa_S[active_adapter]
                    if len(pissa_S.shape)==2:
                        before_V = after_U @ pissa_S
                    else:
                        before_V = after_U * pissa_S
                else:
                    before_V = after_U
                    
                result = result + pissa_V(before_V) * scaling
            result = result.to(torch_result_dtype)

        return result
                
    def __repr__(self) -> str:
        rep = super().__repr__()
        return "pissa." + rep

class Embedding(nn.Module, PiSSALayer):
    # PiSSA implemented in a Embedding layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        pissa_dropout: float = 0.0,
        init_pissa_weights: Union[bool, str] = True,
        fsvd: int = None,
        **kwargs,
    ) -> None:
        super().__init__()
        PiSSALayer.__init__(self, base_layer)

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            pissa_dropout=pissa_dropout,
            init_pissa_weights=init_pissa_weights,
            fsvd=fsvd
        )

    def update_layer(self, adapter_name, r, pissa_dropout, init_pissa_weights, fsvd):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.fsvd[adapter_name] = fsvd
        # Actual trainable parameters
        self.pissa_embedding_U[adapter_name] = nn.Parameter(torch.randn((self.in_features, r)))
        self.pissa_embedding_S[adapter_name] = nn.Parameter(torch.zeros(r))
        self.pissa_embedding_V[adapter_name] = nn.Parameter(torch.randn((r, self.out_features)))
        if init_pissa_weights:
            self.reset_pissa_parameters(adapter_name)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)
        
    def reset_pissa_parameters(self, adapter_name):
        weight = self.get_base_layer().weight
        dtype = weight.dtype
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            raise TypeError(
                "Please initialize PiSSA under float32, float16, or bfloat16. "
                "Subsequently, re-quantize the residual model to help minimize quantization errors."
            )
        weight = weight.to(torch.float32)
        pissa_U, pissa_S, pissa_V = self.svd_or_fsvd(adapter_name, weight)
        self.pissa_embedding_U[adapter_name].data = pissa_U
        self.pissa_embedding_S[adapter_name].data = pissa_S
        self.pissa_embedding_V[adapter_name].data = pissa_V
        weight = weight.data - pissa_U * pissa_S @ pissa_V
        weight = weight.to(dtype)
        self.get_base_layer().weight.data = weight

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.pissa_embedding_U.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights = orig_weights + self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data = base_layer.weight.data + self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.pissa_embedding_U.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.pissa_embedding_U[adapter].device
        dtype = self.pissa_embedding_U[adapter].dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_U = self.pissa_embedding_U[adapter]
        weight_S = self.pissa_embedding_S[adapter]
        weight_V = self.pissa_embedding_V[adapter]

        if cast_to_fp32:
            weight_U = weight_U.float()
            weight_S = weight_S.float()
            weight_V = weight_V.float()

        output_tensor = weight_U * weight_S @ weight_V

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.pissa_embedding_U[adapter] = weight_U.to(dtype)
            self.pissa_embedding_S[adapter] = weight_S.to(dtype)
            self.pissa_embedding_V[adapter] = weight_V.to(dtype)

        return output_tensor

    def _mixed_batch_forward(
        self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        result = self.base_layer(x, *args, **kwargs)
        
        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.pissa_embedding_U.keys():
                continue

            embedding_U = self.pissa_embedding_U[active_adapter]
            embedding_S = self.pissa_embedding_S[active_adapter]
            embedding_V = self.pissa_embedding_V[active_adapter]
            # getting the sub-batch, passing it to PiSSA layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = x[sub_batch_indices_list[i]]
            after_U = self._embed(sub_batch, embedding_U)
            after_S = after_U * embedding_S
            result[sub_batch_indices_list[i]] += (after_S @ embedding_V)

        return result

    def _embed(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        base_layer = self.get_base_layer()
        return F.embedding(
            input,
            weight,
            padding_idx=base_layer.padding_idx,
            max_norm=base_layer.max_norm,
            norm_type=base_layer.norm_type,
            scale_grad_by_freq=base_layer.scale_grad_by_freq,
            sparse=base_layer.sparse,
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # TODO: no dtype conversion here, unlike in Linear, is that correct?
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
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.pissa_embedding_U:
                    continue
                embedding_U = self.pissa_embedding_U[active_adapter]
                embedding_S = self.pissa_embedding_S[active_adapter]
                embedding_V = self.pissa_embedding_V[active_adapter]
                after_U = self._embed(x, embedding_U)
                after_S = after_U * embedding_S
                result = result + (after_S @ embedding_V)
            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "pissa." + rep

class Conv2d(nn.Module, PiSSALayer):
    # PiSSA implemented in a conv2d layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        pissa_dropout: float = 0.0,
        init_pissa_weights: Union[bool, str] = True,
        fsvd: int = None,
        **kwargs,
    ) -> None:
        super().__init__()
        PiSSALayer.__init__(self, base_layer)

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            pissa_dropout,
            init_pissa_weights=init_pissa_weights,
            fsvd=fsvd
        )

    def update_layer(self, adapter_name, r, pissa_dropout, init_pissa_weights, fsvd):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.fsvd[adapter_name] = fsvd
        if pissa_dropout > 0.0:
            self.pissa_dropout = nn.Dropout(p=pissa_dropout)
        else:
            self.pissa_dropout = nn.Identity()
        # Actual trainable parameters
        base_layer = self.get_base_layer()
        kernel_size = base_layer.kernel_size
        stride = base_layer.stride
        padding = base_layer.padding
        self.pissa_U[adapter_name] = nn.Conv2d(self.in_features, r, kernel_size, stride, padding, bias=False)
        self.pissa_S[adapter_name] = nn.Parameter(torch.zeros(r))
        self.pissa_V[adapter_name] = nn.Conv2d(r, self.out_features, (1, 1), (1, 1), bias=False)

        if init_pissa_weights:
            self.reset_pissa_parameters(adapter_name)

        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        self.set_adapter(self.active_adapters)
        
    def reset_pissa_parameters(self, adapter_name):
        weight = self.get_base_layer().weight.data
        n,c,h,w = weight.shape
        dtype = weight.dtype
        weight = weight.view(n,c*h*w)
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            raise TypeError(
                "Please initialize PiSSA under float32, float16, or bfloat16. "
                "Subsequently, re-quantize the residual model to help minimize quantization errors."
            )
        weight = weight.to(torch.float32)
        pissa_V, pissa_S, pissa_U = self.svd_or_fsvd(adapter_name, weight)
        self.pissa_U[adapter_name].weight.data = pissa_U.view(min(self.r[adapter_name], pissa_S.shape[0]),c,h,w)
        self.pissa_S[adapter_name].data = pissa_S
        self.pissa_V[adapter_name].weight.data = pissa_V[:,:,None,None]
        weight = weight - pissa_V * pissa_S @ pissa_U
        weight = weight.to(dtype)
        self.get_base_layer().weight.data = weight.view(n,c,h,w)
         
    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights inside the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.pissa_U.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)

                    orig_weights = orig_weights + delta_weight

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )
                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    base_layer.weight.data = base_layer.weight.data + delta_weight
            

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.pissa_U.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                weight.data -= delta_weight

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.pissa_V[adapter].weight.device
        dtype = self.pissa_U[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_U = self.pissa_U[adapter].weight
        weight_S = self.pissa_S[adapter]
        weight_V = self.pissa_V[adapter].weight

        if cast_to_fp32:
            weight_U = weight_U.float()
            weight_S = weight_S.float()
            weight_V = weight_V.float()
            
        n,c,h,w = self.get_base_layer().weight.shape
        output_tensor = (weight_V.view(n,self.r[adapter]) * weight_S @ weight_U.view(self.r[adapter],-1)).view(n,c,h,w)

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.pissa_U[adapter].weight.data = weight_U.to(dtype)
            self.pissa_S[adapter].weight.data = weight_S.to(dtype)
            self.pissa_V[adapter].weight.data = weight_V.to(dtype)

        return output_tensor

    def _mixed_batch_forward(
        self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.pissa_U.keys():
                continue

            pissa_U = self.pissa_U[active_adapter]
            pissa_S = self.pissa_S[active_adapter]
            pissa_V = self.pissa_V[active_adapter]

            # getting the sub-batch, passing it to PiSSA layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = x[sub_batch_indices_list[i]].to(pissa_U.weight.dtype)
            after_U = pissa_U(self.pissa_dropout(sub_batch))
            after_S = after_U * pissa_S[None,:,None,None]
            result[sub_batch_indices_list[i]] += pissa_V(after_S).to(torch_result_dtype)

        return result
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
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
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            for active_adapter in self.active_adapters:
                if active_adapter not in self.pissa_U.keys():
                    continue
                pissa_U = self.pissa_U[active_adapter]
                pissa_S = self.pissa_S[active_adapter]
                pissa_V = self.pissa_V[active_adapter]
                x = x.to(pissa_U.weight.dtype)
                after_U = pissa_U(self.pissa_dropout(x))
                after_S = after_U * pissa_S[None,:,None,None]
                result = result + pissa_V(after_S)

            result = result.to(torch_result_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "pissa." + rep

def dispatch_default(
    target: torch.nn.Module,
    adapter_name: str,
    pissa_config: PiSSAConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Embedding):
        embedding_kwargs = kwargs.copy()
        embedding_kwargs.pop("fan_in_fan_out", None)
        new_module = Embedding(target, adapter_name, **embedding_kwargs)
    elif isinstance(target_base_layer, torch.nn.Conv2d):
        new_module = Conv2d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = pissa_config.fan_in_fan_out = False
        new_module = Linear(target, adapter_name, **kwargs)

    return new_module
