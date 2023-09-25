# coding=utf-8
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
from transformers.testing_utils import require_sentencepiece, require_tokenizers, require_mindspore, slow, mindspore_device


if is_mindspore_available():
    import mindspore

    from transformers import CamembertModel


@require_mindspore
@require_sentencepiece
@require_tokenizers
class CamembertModelIntegrationTest(unittest.TestCase):
    @slow
    def test_output_embeds_base_model(self):
        model = CamembertModel.from_pretrained("camembert-base")
        model.to(mindspore_device)

        input_ids = mindspore.tensor(
            [[5, 121, 11, 660, 16, 730, 25543, 110, 83, 6]],
            device=mindspore_device,
            dtype=mindspore.long,
        )  # J'aime le camembert !
        output = model(input_ids)["last_hidden_state"]
        expected_shape = mindspore.Size((1, 10, 768))
        self.assertEqual(output.shape, expected_shape)
        # compare the actual values for a slice.
        expected_slice = mindspore.tensor(
            [[[-0.0254, 0.0235, 0.1027], [0.0606, -0.1811, -0.0418], [-0.1561, -0.1127, 0.2687]]],
            device=mindspore_device,
            dtype=mindspore.float,
        )
        # camembert = mindspore.hub.load('mindspore/fairseq', 'camembert.v0')
        # camembert.eval()
        # expected_slice = roberta.model.forward(input_ids)[0][:, :3, :3].detach()

        self.assertTrue(mindspore.allclose(output[:, :3, :3], expected_slice, atol=1e-4))
