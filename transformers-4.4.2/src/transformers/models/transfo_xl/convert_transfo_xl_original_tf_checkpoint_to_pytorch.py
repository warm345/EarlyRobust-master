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
"""Convert Transformer XL checkpoint and datasets."""


import argparse
import os
import pickle
import sys

import mindspore

from transformers import TransfoXLConfig, TransfoXLLMHeadModel, load_tf_weights_in_transfo_xl
from transformers.file_utils import CONFIG_NAME, WEIGHTS_NAME
from transformers.models.transfo_xl import tokenization_transfo_xl as data_utils
from transformers.models.transfo_xl.tokenization_transfo_xl import CORPUS_NAME, VOCAB_FILES_NAMES
from transformers.utils import logging


logging.set_verbosity_info()

# We do this to be able to load python 2 datasets pickles
# See e.g. https://stackoverflow.com/questions/2121874/python-pickling-after-changing-a-modules-directory/2121918#2121918
data_utils.Vocab = data_utils.TransfoXLTokenizer
data_utils.Corpus = data_utils.TransfoXLCorpus
sys.modules["data_utils"] = data_utils
sys.modules["vocabulary"] = data_utils


def convert_transfo_xl_checkpoint_to_mindspore(
    tf_checkpoint_path, transfo_xl_config_file, mindspore_dump_folder_path, transfo_xl_dataset_file
):
    if transfo_xl_dataset_file:
        # Convert a pre-processed corpus (see original TensorFlow repo)
        with open(transfo_xl_dataset_file, "rb") as fp:
            corpus = pickle.load(fp, encoding="latin1")
        # Save vocabulary and dataset cache as Dictionaries (should be better than pickles for the long-term)
        mindspore_vocab_dump_path = mindspore_dump_folder_path + "/" + VOCAB_FILES_NAMES["pretrained_vocab_file"]
        print("Save vocabulary to {}".format(mindspore_vocab_dump_path))
        corpus_vocab_dict = corpus.vocab.__dict__
        mindspore.save(corpus_vocab_dict, mindspore_vocab_dump_path)

        corpus_dict_no_vocab = corpus.__dict__
        corpus_dict_no_vocab.pop("vocab", None)
        mindspore_dataset_dump_path = mindspore_dump_folder_path + "/" + CORPUS_NAME
        print("Save dataset to {}".format(mindspore_dataset_dump_path))
        mindspore.save(corpus_dict_no_vocab, mindspore_dataset_dump_path)

    if tf_checkpoint_path:
        # Convert a pre-trained TensorFlow model
        config_path = os.path.abspath(transfo_xl_config_file)
        tf_path = os.path.abspath(tf_checkpoint_path)

        print("Converting Transformer XL checkpoint from {} with config at {}".format(tf_path, config_path))
        # Initialise mindspore model
        if transfo_xl_config_file == "":
            config = TransfoXLConfig()
        else:
            config = TransfoXLConfig.from_json_file(transfo_xl_config_file)
        print("Building mindspore model from configuration: {}".format(str(config)))
        model = TransfoXLLMHeadModel(config)

        model = load_tf_weights_in_transfo_xl(model, config, tf_path)
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
    parser.add_argument(
        "--mindspore_dump_folder_path",
        default=None,
        type=str,
        required=True,
        help="Path to the folder to store the mindspore model or dataset/vocab.",
    )
    parser.add_argument(
        "--tf_checkpoint_path",
        default="",
        type=str,
        help="An optional path to a TensorFlow checkpoint path to be converted.",
    )
    parser.add_argument(
        "--transfo_xl_config_file",
        default="",
        type=str,
        help="An optional config json file corresponding to the pre-trained BERT model. \n"
        "This specifies the model architecture.",
    )
    parser.add_argument(
        "--transfo_xl_dataset_file",
        default="",
        type=str,
        help="An optional dataset file to be converted in a vocabulary.",
    )
    args = parser.parse_args()
    convert_transfo_xl_checkpoint_to_mindspore(
        args.tf_checkpoint_path,
        args.transfo_xl_config_file,
        args.mindspore_dump_folder_path,
        args.transfo_xl_dataset_file,
    )
