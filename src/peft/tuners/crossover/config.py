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
from torch import nn
from peft.config import PeftConfig
from peft.utils import PeftType

@dataclass
class CrossoverConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`CrossoverModel`].

    Args:
        target_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to apply the adapter to. If this is specified, only the modules with the specified
            names will be replaced. When passing a string, a regex match will be performed. When passing a list of
            strings, either an exact match will be performed or it is checked if the name of the module ends with any
            of the passed strings. If this is specified as 'all-linear', then all linear/Conv1D modules are chosen,
            excluding the output layer. If this is not specified, modules will be chosen according to the model
            architecture. If the architecture is not known, an error will be raised -- in this case, you should specify
            the target modules manually.
        exclude_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to not apply the adapter. When passing a string, a regex match will be performed.
            When passing a list of strings, either an exact match will be performed or it is checked if the name of the
            module ends with any of the passed strings.
        bias (`str`):
            Bias type for Crossover. Can be 'none', 'all' or 'crossover_only'. If 'all' or 'crossover_only', the corresponding biases
            will be updated during training. Be aware that this means that, even when disabling the adapters, the model
            will not produce the same output as the base model would have without adaptation.
        modules_to_save (`List[str]`):
            List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint.
        init_crossover_weights (`bool` | `Literal["gaussian", "eva", "ocrossover", "pissa", "pissa_niter_[number of iters]", "loftq"]`):
            How to initialize the weights of the adapter layers. Passing True (default) results in the default
            initialization from the reference implementation from Microsoft. Passing 'gaussian' results in Gaussian
            initialization scaled by the Crossover rank for linear and layers. Setting the initialization to False leads to
            completely random initialization and is discouraged. Pass `'loftq'` to use LoftQ initialization. Passing
            `'eva'` results in a data-driven initialization of <ahref='https://arxiv.org/abs/2410.07170' >Explained
            Variance Adaptation</a>. EVA initalizes Crossover based on the SVD of layer input activations and achieves SOTA
            performance due to its ability to adapt to the finetuning data. Pass `'ocrossover'` to use OCrossover initialization.
            Passing `'pissa'` results in the initialization of <ahref='https://arxiv.org/abs/2404.02948' >Principal
            Singular values and Singular vectors Adaptation (PiSSA)</a>, which converges more rapidly than Crossover and
            ultimately achieves superior performance. Moreover, PiSSA reduces the quantization error compared to QCrossover,
            leading to further enhancements. Passing `'pissa_niter_[number of iters]'` initiates Fast-SVD-based PiSSA
            initialization, where `[number of iters]` indicates the number of subspace iterations to perform FSVD, and
            must be a nonnegative integer. When `[number of iters]` is set to 16, it can complete the initialization of
            a 7B model within seconds, and the training effect is approximately equivalent to using SVD.
    """
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with Crossover."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'."
                "This can also be a wildcard 'all-linear' which matches all linear/Conv1D layers except the output layer."
                "If not specified, modules will be chosen according to the model architecture, If the architecture is "
                "not known, an error will be raised -- in this case, you should specify the target modules manually."
            ),
        },
    )
    block_size: Optional[int] = field(
        default=64,
        metadata={
            "help": (
            ),
        },
    )
    alpha: int = field(default=64, metadata={"help": "Crossover alpha"})
    dropout: float = field(default=0.0, metadata={"help": "Crossover dropout"})
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to exclude from Crossover."},
    )
    bias: Literal["none", "all", "crossover_only"] = field(
        default="none", metadata={"help": "Bias type for Crossover. Can be 'none', 'all' or 'crossover_only'"}
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from Crossover layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_crossover_weights: Literal['kaiming', 'gaussian', 'orthogonal'] = field(
        default="kaiming",
        metadata={
            "help": (
                "How to initialize the weights of the Crossover layers. "
                "Passing `'eye'` results in OCrossover initialization."
                "Passing `'svd'` initiates Fast-SVD-based PiSSA initialization, "
                "where [number of iters] indicates the number of subspace iterations to perform fsvd, and must be a nonnegative integer."
                "Pass `'loftq'` to use LoftQ initialization"
            ),
        },
    )

    def to_dict(self):
        """
        Returns the configuration for your adapter model as a dictionary. Removes runtime configurations.
        """
        rv = super().to_dict()
        return rv

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.CROSSOVER
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        self.exclude_modules = (
            set(self.exclude_modules) if isinstance(self.exclude_modules, list) else self.exclude_modules
        )
