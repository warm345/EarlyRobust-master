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
"""Convert BERT checkpoint."""


import argparse
import os

import mindspore

from transformers import (
    XLNetConfig,
    XLNetForQuestionAnswering,
    XLNetForSequenceClassification,
    XLNetLMHeadModel,
    load_tf_weights_in_xlnet,
)
from transformers.file_utils import CONFIG_NAME, WEIGHTS_NAME
from transformers.utils import logging


GLUE_TASKS_NUM_LABELS = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
}


logging.set_verbosity_info()


def convert_xlnet_checkpoint_to_mindspore(
    tf_checkpoint_path, bert_config_file, mindspore_dump_folder_path, finetuning_task=None
):
    # Initialise mindspore model
    config = XLNetConfig.from_json_file(bert_config_file)

    finetuning_task = finetuning_task.lower() if finetuning_task is not None else ""
    if finetuning_task in GLUE_TASKS_NUM_LABELS:
        print("Building mindspore XLNetForSequenceClassification model from configuration: {}".format(str(config)))
        config.finetuning_task = finetuning_task
        config.num_labels = GLUE_TASKS_NUM_LABELS[finetuning_task]
        model = XLNetForSequenceClassification(config)
    elif "squad" in finetuning_task:
        config.finetuning_task = finetuning_task
        model = XLNetForQuestionAnswering(config)
    else:
        model = XLNetLMHeadModel(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_xlnet(model, config, tf_checkpoint_path)

    # Save mindspore-model
    mindspore_weights_dump_path = os.path.join(mindspore_dump_folder_path, WEIGHTS_NAME)
    mindspore_config_dump_path = os.path.join(mindspore_dump_folder_path, CONFIG_NAME)
    print("Save mindspore model to {}".format(os.path.abspath(mindspore_weights_dump_path)))
    mindspore.save(model.state_dict(), mindspore_weights_dump_path)
    print("Save configuration file to {}".format(os.path.abspath(mindspore_config_dump_path)))
    with open(mindspore_config_dump_path, "w", encoding="utf-8") as f:
        f.write(config.to_json_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--xlnet_config_file",
        default=None,
        type=str,
        required=True,
        help="The config json file corresponding to the pre-trained XLNet model. \n"
        "This specifies the model architecture.",
    )
    parser.add_argument(
        "--mindspore_dump_folder_path",
        default=None,
        type=str,
        required=True,
        help="Path to the folder to store the mindspore model or dataset/vocab.",
    )
    parser.add_argument(
        "--finetuning_task",
        default=None,
        type=str,
        help="Name of a task on which the XLNet TensorFlow model was fine-tuned",
    )
    args = parser.parse_args()
    print(args)

    convert_xlnet_checkpoint_to_mindspore(
        args.tf_checkpoint_path, args.xlnet_config_file, args.mindspore_dump_folder_path, args.finetuning_task
    )
