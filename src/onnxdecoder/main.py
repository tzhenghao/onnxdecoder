# Standard imports
import json
import logging
from pathlib import Path
from typing import Any

# Third party imports
from pydantic.dataclasses import dataclass
import click
import onnx

DEFAULT_INPUT_ONNX_FILE_PATH = "../../assets/resnet50-v2-7.onnx"
DEFAULT_OUTPUT_JSON_DIRECTORY = "outputs"
DEFAULT_JSON_INDENT_SIZE = 2

logger = logging.getLogger(__name__)


@dataclass
class FlatOnnx:
    onnx_graph_nodes: list[dict[str, Any]]
    onnx_inputs: dict[str, Any]
    onnx_outputs: dict[str, Any]


@dataclass
class CLIConfig:
    input_onnx_file: str
    output_directory: str
    enable_output_json: bool
    json_indent_size: int


@dataclass
class CLIContext:
    onnx_model: Any
    flat_onnx: FlatOnnx | None = None


@click.command()
@click.argument("input_onnx_file")
@click.option(
    "--output_directory",
    default=DEFAULT_OUTPUT_JSON_DIRECTORY,
    help="The output directory where all JSON files generated will be placed",
)
@click.option(
    "--enable_output_json",
    default=True,
    help="Enable output JSONs",
)
@click.option(
    "--json_indent_size",
    default=DEFAULT_JSON_INDENT_SIZE,
    help="JSON indent size, defaults to {default_json_indent_size}".format(
        default_json_indent_size=DEFAULT_JSON_INDENT_SIZE
    ),
)
def cli(
    input_onnx_file,
    output_directory,
    enable_output_json,
    json_indent_size,
):
    onnx_model = onnx.load(DEFAULT_INPUT_ONNX_FILE_PATH)
    click.secho(
        "IR version: {ir_version}".format(ir_version=onnx_model.ir_version),
        fg="yellow",
    )
    cli_config = CLIConfig(
        input_onnx_file=input_onnx_file,
        output_directory=output_directory,
        enable_output_json=enable_output_json,
        json_indent_size=json_indent_size,
    )
    cli_context = CLIContext(
        onnx_model=onnx_model,
    )

    parse_onnx_graph(cli_context=cli_context)

    generate_outputs(cli_config=cli_config, cli_context=cli_context)


def parse_onnx_graph(cli_context: CLIContext):
    onnx_graph_nodes = []
    onnx_name_to_inputs: dict[str, Any] = {}
    onnx_name_to_outputs: dict[str, Any] = {}

    for node in cli_context.onnx_model.graph.node:
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
    cli_context.flat_onnx = FlatOnnx(
        onnx_graph_nodes=onnx_graph_nodes,
        onnx_inputs=onnx_name_to_inputs,
        onnx_outputs=onnx_name_to_outputs,
    )


def generate_outputs(cli_config: CLIConfig, cli_context: CLIContext):
    if cli_config.enable_output_json:
        msg = "enable_output_json is enabled, printing output to {output_directory}...".format(  # noqa: E501
            output_directory=DEFAULT_OUTPUT_JSON_DIRECTORY
        )
        click.secho(
            msg,
            fg="green",
        )

        output_path = Path(DEFAULT_OUTPUT_JSON_DIRECTORY).resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        with open(
            output_path.joinpath("output-model-outputs.json"), "w"
        ) as output_file:
            json.dump(
                cli_context.flat_onnx.onnx_outputs,
                output_file,
                indent=cli_config.json_indent_size,
            )

        with open(
            output_path.joinpath("output-model-inputs.json"), "w"
        ) as output_file:
            json.dump(
                cli_context.flat_onnx.onnx_inputs,
                output_file,
                indent=cli_config.json_indent_size,
            )

        with open(
            output_path.joinpath("output-model-nodes.json"), "w"
        ) as output_file:
            json.dump(
                cli_context.flat_onnx.onnx_graph_nodes,
                output_file,
                indent=cli_config.json_indent_size,
            )


if __name__ == "__main__":
    cli()
