# Standard imports
import json
import logging
import os
from copy import deepcopy
from pathlib import Path

# Third party imports
import click
import onnx
from onnx.external_data_helper import convert_model_to_external_data
from google.protobuf.json_format import MessageToDict
from pydantic.dataclasses import dataclass

enable_output_files = True
INPUT_ONNX_FILE_PATH = "original-resnet50-v2-7.onnx"
OUTPUT_ONNX_FILE_PATH = "new-resnet50-v2-7.onnx"
INDENT_SIZE = 2

logger = logging.getLogger(__name__)


def main():
    onnx_model = onnx.load(INPUT_ONNX_FILE_PATH)

    onnx.save_model(
        onnx_model,
        OUTPUT_ONNX_FILE_PATH,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="filename",
        size_threshold=0,  # Always convert all to external_data.
        convert_attribute=False,
    )

    # convert_model_to_external_data(
    #     onnx_model,
    #     all_tensors_to_one_file=True,
    #     location="filename",
    #     size_threshold=0,
    #     convert_attribute=False,
    # )
    # # Must be followed by save_model to save the converted model to a specific path
    # onnx.save_model(onnx_model, OUTPUT_ONNX_FILE_PATH)


if __name__ == "__main__":
    main()
