# Standard imports
import json
import logging
import os

# Third party imports
import click
import onnx
from pydantic.dataclasses import dataclass
from google.protobuf.json_format import MessageToJson

enable_output_file = True
INPUT_ONNX_FILE_PATH = "resnet50-v2-7.onnx"
OUTPUT_JSON_PATH = "output.json"
INDENT_SIZE = 2

logger = logging.getLogger(__name__)


def main():
    onnx_graph = onnx.load(INPUT_ONNX_FILE_PATH)
    s = MessageToJson(onnx_graph, preserving_proto_field_name=True)
    onnx_json = json.loads(s)

    logger.info(onnx_json)

    if enable_output_file:
        msg = "enable_output_file is enabled, printing output to {output_file}...".format(  # noqa: E501
            output_file=OUTPUT_JSON_PATH
        )
        click.secho(
            msg,
            fg="green",
        )
        with open(OUTPUT_JSON_PATH, "w") as f:
            json.dump(onnx_json, f, indent=INDENT_SIZE)


if __name__ == "__main__":
    main()
