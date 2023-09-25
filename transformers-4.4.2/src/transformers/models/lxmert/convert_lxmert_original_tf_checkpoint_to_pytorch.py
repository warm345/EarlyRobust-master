# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
"""Convert LXMERT checkpoint."""


import argparse

import mindspore

from transformers import LxmertConfig, LxmertForPreTraining, load_tf_weights_in_lxmert
from transformers.utils import logging


logging.set_verbosity_info()


def convert_tf_checkpoint_to_mindspore(tf_checkpoint_path, config_file, mindspore_dump_path):
    # Initialise mindspore model
    config = LxmertConfig.from_json_file(config_file)
    print("Building mindspore model from configuration: {}".format(str(config)))
    model = LxmertForPreTraining(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_lxmert(model, config, tf_checkpoint_path)

    # Save mindspore-model
    print("Save mindspore model to {}".format(mindspore_dump_path))
    mindspore.save(model.state_dict(), mindspore_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        required=True,
        help="The config json file corresponding to the pre-trained model. \n"
        "This specifies the model architecture.",
    )
    parser.add_argument(
        "--mindspore_dump_path", default=None, type=str, required=True, help="Path to the output mindspore model."
    )
    args = parser.parse_args()
    convert_tf_checkpoint_to_mindspore(args.tf_checkpoint_path, args.config_file, args.mindspore_dump_path)
