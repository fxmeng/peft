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

from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType

@dataclass
class PiSSAConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`PiSSAModel`].

    Args:
        r (`int`):
            PiSSA attention dimension (the "rank").
        pissa_alpha (`int`):
            The alpha parameter for pissa scaling.
        pissa_dropout (`float`):
            The dropout probability for pissa layers.
        singular_value: Optional[str]:
            How to deal with the singular_value. `to_u` or `to_v` will merge it to the left/right singular vector."
            "`to_uv`: will merge the square root of S to both U and V."
            "`vector`: preserve the singular value as a vector for finetuning."
            "`matrix`: preserve the singular value in matrix format for finetuning."
        target_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to apply the adapter to. If this is specified, only the modules with the specified
            names will be replaced. When passing a string, a regex match will be performed. When passing a list of
            strings, either an exact match will be performed or it is checked if the name of the module ends with any
            of the passed strings. If this is specified as 'all-linear', then all linear/Conv1D modules are chosen,
            excluding the output layer. If this is not specified, modules will be chosen according to the model
            architecture. If the architecture is not known, an error will be raised -- in this case, you should specify
            the target modules manually.
        use_rspissa (`bool`):
            When set to True, uses <a href='https://doi.org/10.48550/arXiv.2312.03732'>Rank-Stabilized LoRA</a> which
            sets the adapter scaling factor to `pissa_alpha/math.sqrt(r)`, since it was proven to work better.
            Otherwise, it will use the original default value of `pissa_alpha/r`.
        fan_in_fan_out (`bool`):
            Set this to True if the layer to replace stores weight like (fan_in, fan_out). For example, gpt-2 uses
            `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.
        bias (`str`):
            Bias type for PiSSA. Can be 'none', 'all' or 'pissa_only'. If 'all' or 'pissa_only', the corresponding biases
            will be updated during training. Be aware that this means that, even when disabling the adapters, the model
            will not produce the same output as the base model would have without adaptation.
        modules_to_save (`List[str]`):
            List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint.
        init_pissa_weights (`bool`):
            How to initialize the weights of the adapter layers. Passing True (default) results in the default
            initialization from the reference implementation from PiSSA. Setting the initialization to False leads to
            completely random initialization and is discouraged. For more information, see <a
            href='https://arxiv.org/abs/2404.02948'>Principal Singular values and Singular vectors Adaptation</a>.
        layers_to_transform (`Union[List[int], int]`):
            The layer indices to transform. If a list of ints is passed, it will apply the adapter to the layer indices
            that are specified in this list. If a single integer is passed, it will apply the transformations on the
            layer at this index.
        layers_pattern (`str`):
            The layer pattern name, used only if `layers_to_transform` is different from `None`.
        rank_pattern (`dict`):
            The mapping from layer names or regexp expression to ranks which are different from the default rank
            specified by `r`.
        megatron_config (`Optional[dict]`):
            The TransformerConfig arguments for Megatron. It is used to create PiSSA's parallel linear layer. You can
            get it like this, `core_transformer_config_from_args(get_args())`, these two functions being from Megatron.
            The arguments will be used to initialize the TransformerConfig of Megatron. You need to specify this
            parameter when you want to apply PiSSA to the ColumnParallelLinear and RowParallelLinear layers of megatron.
        megatron_core (`Optional[str]`):
            The core module from Megatron to use, defaults to `"megatron.core"`.
        layer_replication (`List[Tuple[int, int]]`):
            Build a new stack of layers by stacking the original model layers according to the ranges specified. This
            allows expanding (or shrinking) the model without duplicating the base model weights. The new layers will
            all have separate PiSSA adapters attached to them.
    """

    r: int = field(default=8, metadata={"help": "PiSSA rank"})
    pissa_alpha: int = field(default=8, metadata={"help": "PiSSA alpha"})
    pissa_dropout: float = field(default=0.0, metadata={"help": "PiSSA dropout"})
    singular_value: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "`None`: merge the square root of the singular value to both singular vector."
                "`vector`: preserve the singular value as a vector for finetuning."
                "`matrix`: preserve the singular value in matrix format for finetuning."
            )
        },
    )
    freeze: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "`USVA`"
                "When the input contain `U`, will freeze the left singular vector."
                "When the input contain `V`, will freeze the right singular vector."
                "When the input contain `S`, will freeze the singular value."
                "When the input contain `A`, will freeze one of U and V which has more number of parameters."
            )
        },
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with PiSSA."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'."
                "This can also be a wildcard 'all-linear' which matches all linear/Conv1D layers except the output layer."
                "If not specified, modules will be chosen according to the model architecture, If the architecture is "
                "not known, an error will be raised -- in this case, you should specify the target modules manually."
            ),
        },
    )
    use_rspissa: bool = field(
        default=False,
        metadata={
            "help": (
                "When set to True, uses <a href='https://doi.org/10.48550/arXiv.2312.03732'>Rank-Stabilized LoRA</a>"
                " which sets the adapter scaling factor to `pissa_alpha/math.sqrt(r)`, since it"
                " was proven to work better. Otherwise, it will use the original default"
                " value of `pissa_alpha/r`."
            )
        },
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: Literal["none", "all", "pissa_only"] = field(
        default="none", metadata={"help": "Bias type for PiSSA. Can be 'none', 'all' or 'pissa_only'"}
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from PiSSA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_pissa_weights: bool = field(
        default=False,
        metadata={
            "help": (
                "How to initialize the weights of the PiSSA layers. Passing True results in PiSSA initialization."
                "Passing False (default) results in the random initialization."
            ),
        },
    )
    fsvd : int = field(
        default=None,
        metadata={
            "help": (
                "Fast-SVD-based PiSSA initialization, where [fsvd] indicates the number of subspace iterations to perform fsvd, and must be a nonnegative integer."
            ),
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index. "
            "This only works when target_modules is a list of str."
        },
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern."
            "This only works when target_modules is a list of str."
        },
    )
    rank_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to ranks which are different from the default rank specified by `r`. "
                "For example, `{model.decoder.layers.0.encoder_attn.k_proj: 8`}"
            )
        },
    )
    alpha_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to alphas which are different from the default alpha specified by `pissa_alpha`. "
                "For example, `{model.decoder.layers.0.encoder_attn.k_proj: 32`}"
            )
        },
    )
    megatron_config: Optional[dict] = field(
        default=None,
        metadata={
            "help": (
                "The TransformerConfig from Megatron. It is used to create PiSSA's parallel linear layer."
                "You can get it like this, `core_transformer_config_from_args(get_args())`, "
                "these two functions being from Megatron."
                "You need to specify this parameter when you want to apply PiSSA to the ColumnParallelLinear and "
                "RowParallelLinear layers of megatron."
                "It should be noted that we may not be able to use the `save_pretrained` and `from_pretrained` "
                "functions, because TransformerConfig may not necessarily be serialized."
                "But when using megatron, we can use `get_peft_model_state_dict` function and "
                "megatron's framework, they can also save and load models and configurations."
            )
        },
    )
    megatron_core: Optional[str] = field(
        default="megatron.core",
        metadata={
            "help": (
                "The core module from Megatron, it is used to create PiSSA's parallel linear layer. "
                "It only needs to be passed in when you need to use your own modified megatron core module. "
                "Otherwise, it will use the default value `megatron.core`. "
            )
        },
    )
    # Enables replicating layers in a model to expand it to a larger model.
    layer_replication: Optional[list[tuple[int, int]]] = field(
        default=None,
        metadata={
            "help": (
                "This enables using PiSSA to effectively expand a transformer model to a larger size by repeating some layers. "
                "The transformation handles models (currently Llama, Bert or Falcon compatible architectures) with "
                "a module list in the model which it modifies to expand the number of modules. "
                "Base weights are shared so the memory usage is close to the original model. The intended use is these base weights "
                "remain fixed during finetuning but each layer has a separate PiSSA adapter so the layers can be specialed via "
                "the adapter layers fit during fine tuning."
                "The format is a list of [start, end) pairs which specify the layer ranges to stack. For example:\n"
                "   Original model has 5 layers labelled by their position in the model: `[0, 1, 2, 3, 4]`\n"
                "   layer_replication: `[[0, 4], [2, 5]]`\n"
                "   Final model will have this arrangement of original layers: `[0, 1, 2, 3, 2, 3, 4]`\n"
                "This format is based on what is used for pass-through merges in mergekit. It makes it simple to select sequential "
                "ranges of a model and stack them while reusing layers at either end of each sequence."
            )
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.PiSSA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        # if target_modules is a regex expression, then layers_to_transform should be None
        if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
            raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")

        # if target_modules is a regex expression, then layers_pattern should be None
        if isinstance(self.target_modules, str) and self.layers_pattern is not None:
            raise ValueError("`layers_pattern` cannot be used when `target_modules` is a str.")
        
        if self.singular_value not in [None,"vector","matrix"]:
            raise ValueError("`singular_value` must be one of [None,`vector`,`matrix`]")