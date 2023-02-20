# Standard imports
import json
import logging
import os
from copy import deepcopy
from pathlib import Path

# Third party imports
import click
import onnx
from google.protobuf.json_format import MessageToDict
from pydantic.dataclasses import dataclass

enable_output_files = True
INPUT_ONNX_FILE_PATH = "resnet50-v2-7.onnx"
OUTPUT_JSON_DIRECTORY = "."
OUTPUT_JSON_PATH = "output2.json"
INDENT_SIZE = 2

logger = logging.getLogger(__name__)


def main():
    onnx_graph = onnx.load(INPUT_ONNX_FILE_PATH)
    onnx_dict = MessageToDict(onnx_graph, preserving_proto_field_name=True)
    # onnx_json = json.loads(s)

    onnx_metadata = deepcopy(onnx_dict)
    del onnx_metadata["graph"]

    click.secho("ONNX metadata: {}".format(onnx_metadata), fg="yellow")
    logger.debug("ONNX dict: {onnx_dict}".format(onnx_dict=onnx_dict))

    if enable_output_files:
        msg = "enable_output_files is enabled, printing output to {output_directory}...".format(  # noqa: E501
            output_directory=OUTPUT_JSON_DIRECTORY
        )
        click.secho(
            msg,
            fg="green",
        )

        output_path = Path(OUTPUT_JSON_DIRECTORY).resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        with open(output_path.joinpath("output-metadata.json"), "w") as f:
            json.dump(onnx_dict, f, indent=INDENT_SIZE)

        with open(output_path.joinpath("output-all.json"), "w") as f:
            json.dump(onnx_dict, f, indent=INDENT_SIZE)


if __name__ == "__main__":
    main()
