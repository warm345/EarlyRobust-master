# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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

import argparse
import os
import re


# All paths are set with the intent you should run this script from the root of the repo with the command
# python utils/check_dummies.py
PATH_TO_TRANSFORMERS = "src/transformers"

_re_single_line_import = re.compile(r"\s+from\s+\S*\s+import\s+([^\(\s].*)\n")
_re_test_backend = re.compile(r"^\s+if\s+is\_([a-z]*)\_available\(\):\s*$")


BACKENDS = ["mindspore", "tf", "flax", "sentencepiece", "tokenizers"]


DUMMY_CONSTANT = """
{0} = None
"""

DUMMY_PRETRAINED_CLASS = """
class {0}:
    def __init__(self, *args, **kwargs):
        requires_{1}(self)

    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_{1}(self)
"""

DUMMY_CLASS = """
class {0}:
    def __init__(self, *args, **kwargs):
        requires_{1}(self)
"""

DUMMY_FUNCTION = """
def {0}(*args, **kwargs):
    requires_{1}({0})
"""


def read_init():
    """ Read the init and extracts mindspore, TensorFlow, SentencePiece and Tokenizers objects. """
    with open(os.path.join(PATH_TO_TRANSFORMERS, "__init__.py"), "r", encoding="utf-8", newline="\n") as f:
        lines = f.readlines()

    # Get to the point we do the actual imports for type checking
    line_index = 0
    while not lines[line_index].startswith("if TYPE_CHECKING"):
        line_index += 1

    backend_specific_objects = {}
    # Go through the end of the file
    while line_index < len(lines):
        # If the line is an if is_backemd_available, we grab all objects associated.
        if _re_test_backend.search(lines[line_index]) is not None:
            backend = _re_test_backend.search(lines[line_index]).groups()[0]
            line_index += 1

            # Ignore if backend isn't tracked for dummies.
            if backend not in BACKENDS:
                continue

            objects = []
            # Until we unindent, add backend objects to the list
            while len(lines[line_index]) <= 1 or lines[line_index].startswith(" " * 8):
                line = lines[line_index]
                single_line_import_search = _re_single_line_import.search(line)
                if single_line_import_search is not None:
                    objects.extend(single_line_import_search.groups()[0].split(", "))
                elif line.startswith(" " * 12):
                    objects.append(line[12:-2])
                line_index += 1

            backend_specific_objects[backend] = objects
        else:
            line_index += 1

    return backend_specific_objects


def create_dummy_object(name, backend_name):
    """ Create the code for the dummy object corresponding to `name`."""
    _pretrained = [
        "Config" "ForCausalLM",
        "ForConditionalGeneration",
        "ForMaskedLM",
        "ForMultipleChoice",
        "ForQuestionAnswering",
        "ForSequenceClassification",
        "ForTokenClassification",
        "Model",
        "Tokenizer",
    ]
    if name.isupper():
        return DUMMY_CONSTANT.format(name)
    elif name.islower():
        return DUMMY_FUNCTION.format(name, backend_name)
    else:
        is_pretrained = False
        for part in _pretrained:
            if part in name:
                is_pretrained = True
                break
        if is_pretrained:
            return DUMMY_PRETRAINED_CLASS.format(name, backend_name)
        else:
            return DUMMY_CLASS.format(name, backend_name)


def create_dummy_files():
    """ Create the content of the dummy files. """
    backend_specific_objects = read_init()
    # For special correspondence backend to module name as used in the function requires_modulename
    module_names = {"mindspore": "mindspore"}
    dummy_files = {}

    for backend, objects in backend_specific_objects.items():
        backend_name = module_names.get(backend, backend)
        dummy_file = "# This file is autogenerated by the command `make fix-copies`, do not edit.\n"
        dummy_file += f"from ..file_utils import requires_{backend_name}\n\n"
        dummy_file += "\n".join([create_dummy_object(o, backend_name) for o in objects])
        dummy_files[backend] = dummy_file

    return dummy_files


def check_dummies(overwrite=False):
    """ Check if the dummy files are up to date and maybe `overwrite` with the right content. """
    dummy_files = create_dummy_files()
    # For special correspondence backend to shortcut as used in utils/dummy_xxx_objects.py
    short_names = {"mindspore": "pt"}

    # Locate actual dummy modules and read their content.
    path = os.path.join(PATH_TO_TRANSFORMERS, "utils")
    dummy_file_paths = {
        backend: os.path.join(path, f"dummy_{short_names.get(backend, backend)}_objects.py")
        for backend in dummy_files.keys()
    }

    actual_dummies = {}
    for backend, file_path in dummy_file_paths.items():
        with open(file_path, "r", encoding="utf-8", newline="\n") as f:
            actual_dummies[backend] = f.read()

    for backend in dummy_files.keys():
        if dummy_files[backend] != actual_dummies[backend]:
            if overwrite:
                print(
                    f"Updating transformers.utils.dummy_{short_names.get(backend, backend)}_objects.py as the main "
                    "__init__ has new objects."
                )
                with open(dummy_file_paths[backend], "w", encoding="utf-8", newline="\n") as f:
                    f.write(dummy_files[backend])
            else:
                raise ValueError(
                    "The main __init__ has objects that are not present in "
                    f"transformers.utils.dummy_{short_names.get(backend, backend)}_objects.py. Run `make fix-copies` "
                    "to fix this."
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix_and_overwrite", action="store_true", help="Whether to fix inconsistencies.")
    args = parser.parse_args()

    check_dummies(args.fix_and_overwrite)
