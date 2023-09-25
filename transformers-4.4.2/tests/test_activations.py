# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import unittest

from transformers import is_mindspore_available
from transformers.testing_utils import require_mindspore


if is_mindspore_available():
    import mindspore

    from transformers.activations import _gelu_python, gelu_new, get_activation


@require_mindspore
class TestActivations(unittest.TestCase):
    def test_gelu_versions(self):
        x = mindspore.Tensor([-100, -1, -0.1, 0, 0.1, 1.0, 100])
        mindspore_builtin = get_activation("gelu")
        self.assertTrue(mindspore.eq(_gelu_python(x), mindspore_builtin(x)).all().item())
        self.assertFalse(mindspore.eq(_gelu_python(x), gelu_new(x)).all().item())

    def test_get_activation(self):
        get_activation("swish")
        get_activation("silu")
        get_activation("relu")
        get_activation("tanh")
        get_activation("gelu_new")
        get_activation("gelu_fast")
        with self.assertRaises(KeyError):
            get_activation("bogus")
        with self.assertRaises(KeyError):
            get_activation(None)
