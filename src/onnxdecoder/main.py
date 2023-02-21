# Standard imports
import json
import logging
from pathlib import Path
from typing import Any

# Third party imports
import click
import onnx

enable_output_files = True
INPUT_ONNX_FILE_PATH = "../../assets/resnet50-v2-7.onnx"
OUTPUT_JSON_DIRECTORY = "outputs"
INDENT_SIZE = 2

logger = logging.getLogger(__name__)


def main():
    onnx_model = onnx.load(INPUT_ONNX_FILE_PATH)

    click.secho(
        "IR version: {ir_version}".format(ir_version=onnx_model.ir_version),
        fg="yellow",
    )

    onnx_graph_nodes = []
    onnx_name_to_inputs: dict[str, Any] = {}
    onnx_name_to_outputs: dict[str, Any] = {}

    for node in onnx_model.graph.node:
        click.secho("name: {name}".format(name=node.name), fg="yellow")
        click.secho(
            "op_type: {op_type}".format(op_type=node.op_type), fg="yellow"
        )
        click.secho("input: {input}".format(input=node.input), fg="yellow")
        click.secho("output: {output}".format(output=node.output), fg="yellow")

        onnx_graph_nodes.append({"name": node.name, "op_type": node.op_type})
        onnx_name_to_inputs[node.name] = [
            input_val for input_val in node.input
        ]
        onnx_name_to_outputs[node.name] = [
            output_val for output_val in node.output
        ]

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

        with open(
            output_path.joinpath("output-model-outputs.json"), "w"
        ) as output_file:
            json.dump(onnx_name_to_outputs, output_file, indent=INDENT_SIZE)

        with open(
            output_path.joinpath("output-model-inputs.json"), "w"
        ) as output_file:
            json.dump(onnx_name_to_inputs, output_file, indent=INDENT_SIZE)

        with open(
            output_path.joinpath("output-model-nodes.json"), "w"
        ) as output_file:
            json.dump(onnx_graph_nodes, output_file, indent=INDENT_SIZE)


if __name__ == "__main__":
    main()
