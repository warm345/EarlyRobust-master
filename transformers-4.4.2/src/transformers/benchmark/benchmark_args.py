# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass, field
from typing import Tuple

from ..file_utils import cached_property, is_mindspore_available, is_mindspore_tpu_available, mindspore_required
from ..utils import logging
from .benchmark_args_utils import BenchmarkArguments


if is_mindspore_available():
    import mindspore

if is_mindspore_tpu_available():
    import mindspore_xla.core.xla_model as xm


logger = logging.get_logger(__name__)


@dataclass
class mindsporeBenchmarkArguments(BenchmarkArguments):

    deprecated_args = [
        "no_inference",
        "no_cuda",
        "no_tpu",
        "no_speed",
        "no_memory",
        "no_env_print",
        "no_multi_process",
    ]

    def __init__(self, **kwargs):
        """
        This __init__ is there for legacy code. When removing deprecated args completely, the class can simply be
        deleted
        """
        for deprecated_arg in self.deprecated_args:
            if deprecated_arg in kwargs:
                positive_arg = deprecated_arg[3:]
                setattr(self, positive_arg, not kwargs.pop(deprecated_arg))
                logger.warning(
                    f"{deprecated_arg} is depreciated. Please use --no_{positive_arg} or {positive_arg}={kwargs[positive_arg]}"
                )

        self.mindsporescript = kwargs.pop("mindsporescript", self.mindsporescript)
        self.mindspore_xla_tpu_print_metrics = kwargs.pop("mindspore_xla_tpu_print_metrics", self.mindspore_xla_tpu_print_metrics)
        self.fp16_opt_level = kwargs.pop("fp16_opt_level", self.fp16_opt_level)
        super().__init__(**kwargs)

    mindsporescript: bool = field(default=False, metadata={"help": "Trace the models using mindsporescript"})
    mindspore_xla_tpu_print_metrics: bool = field(default=False, metadata={"help": "Print Xla/mindspore tpu metrics"})
    fp16_opt_level: str = field(
        default="O1",
        metadata={
            "help": (
                "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                "See details at https://nvidia.github.io/apex/amp.html"
            )
        },
    )

    @cached_property
    @mindspore_required
    def _setup_devices(self) -> Tuple["mindspore.device", int]:
        logger.info("mindspore: setting up devices")
        if not self.cuda:
            device = mindspore.device("cpu")
            n_gpu = 0
        elif is_mindspore_tpu_available():
            device = xm.xla_device()
            n_gpu = 0
        else:
            device = mindspore.device("cuda" if mindspore.cuda.is_available() else "cpu")
            n_gpu = mindspore.cuda.device_count()
        return device, n_gpu

    @property
    def is_tpu(self):
        return is_mindspore_tpu_available() and self.tpu

    @property
    @mindspore_required
    def device_idx(self) -> int:
        # TODO(PVP): currently only single GPU is supported
        return mindspore.cuda.current_device()

    @property
    @mindspore_required
    def device(self) -> "mindspore.device":
        return self._setup_devices[0]

    @property
    @mindspore_required
    def n_gpu(self):
        return self._setup_devices[1]

    @property
    def is_gpu(self):
        return self.n_gpu > 0
