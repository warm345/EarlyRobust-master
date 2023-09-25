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
"""Convert OpenAI GPT checkpoint."""


import argparse

import mindspore

from transformers import GPT2Config, GPT2Model, load_tf_weights_in_gpt2
from transformers.file_utils import CONFIG_NAME, WEIGHTS_NAME
from transformers.utils import logging


logging.set_verbosity_info()


def convert_gpt2_checkpoint_to_mindspore(gpt2_checkpoint_path, gpt2_config_file, mindspore_dump_folder_path):
    # Construct model
    if gpt2_config_file == "":
        config = GPT2Config()
    else:
        config = GPT2Config.from_json_file(gpt2_config_file)
    model = GPT2Model(config)

    # Load weights from numpy
    load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path)

    # Save mindspore-model
    mindspore_weights_dump_path = mindspore_dump_folder_path + "/" + WEIGHTS_NAME
    mindspore_config_dump_path = mindspore_dump_folder_path + "/" + CONFIG_NAME
    print("Save mindspore model to {}".format(mindspore_weights_dump_path))
    mindspore.save(model.state_dict(), mindspore_weights_dump_path)
    print("Save configuration file to {}".format(mindspore_config_dump_path))
    with open(mindspore_config_dump_path, "w", encoding="utf-8") as f:
        f.write(config.to_json_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--gpt2_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--mindspore_dump_folder_path", default=None, type=str, required=True, help="Path to the output mindspore model."
    )
    parser.add_argument(
        "--gpt2_config_file",
        default="",
        type=str,
        help="An optional config json file corresponding to the pre-trained OpenAI model. \n"
        "This specifies the model architecture.",
    )
    args = parser.parse_args()
    convert_gpt2_checkpoint_to_mindspore(args.gpt2_checkpoint_path, args.gpt2_config_file, args.mindspore_dump_folder_path)
