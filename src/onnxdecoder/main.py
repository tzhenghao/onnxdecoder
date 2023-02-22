# Standard imports
import json
import logging
from pathlib import Path
from typing import Any

# Third party imports
import click
import onnx
from pydantic.dataclasses import dataclass

DEFAULT_INPUT_ONNX_FILE_PATH = "../../assets/resnet50-v2-7.onnx"
DEFAULT_OUTPUT_JSON_DIRECTORY = "outputs"
DEFAULT_JSON_INDENT_SIZE = 2

logger = logging.getLogger(__name__)


@dataclass
class FlatOnnx:
    onnx_graph_node_names_list: list[str]
    onnx_graph_node_name_to_attributes: dict[str, Any]
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
    onnx_model = onnx.load(input_onnx_file)
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
    onnx_graph_node_names_list = []
    onnx_name_to_inputs: dict[str, Any] = {}
    onnx_name_to_outputs: dict[str, Any] = {}
    onnx_graph_node_name_to_attributes: dict[str, Any] = {}

    for node in cli_context.onnx_model.graph.node:
        click.secho("name: {name}".format(name=node.name), fg="yellow")
        click.secho(
            "op_type: {op_type}".format(op_type=node.op_type), fg="yellow"
        )
        click.secho("input: {input}".format(input=node.input), fg="yellow")
        click.secho("output: {output}".format(output=node.output), fg="yellow")

        onnx_graph_node_names_list.append(node.name)
        onnx_graph_node_name_to_attributes[node.name] = {
            "op_type": node.op_type
        }
        onnx_name_to_inputs[node.name] = [
            input_val for input_val in node.input
        ]
        onnx_name_to_outputs[node.name] = [
            output_val for output_val in node.output
        ]
    cli_context.flat_onnx = FlatOnnx(
        onnx_graph_node_names_list=onnx_graph_node_names_list,
        onnx_graph_node_name_to_attributes=onnx_graph_node_name_to_attributes,
        onnx_inputs=onnx_name_to_inputs,
        onnx_outputs=onnx_name_to_outputs,
    )
    rebuild_nested_onnx_graph_representation(cli_context=cli_context)


def rebuild_nested_onnx_graph_representation(cli_context: CLIContext):
    logger.info("Remapping to nested graph representation...")

    input_to_output_node_dict: dict[str, Any] = {}
    for (
        node_name,
        node_input_list,
    ) in cli_context.flat_onnx.onnx_inputs.items():
        for input_node in node_input_list:
            # If the input_node is not already in the nested graph, we create it
            if input_node not in input_to_output_node_dict:
                input_to_output_node_dict[input_node] = [node_name]
            else:
                # Append for ones that already exist.
                input_to_output_node_dict[input_node].append(node_name)

    with open("input_to_output_node_dict.json", "w") as output_file:
        json.dump(
            input_to_output_node_dict,
            output_file,
            indent=2,
        )

    seen_nodes = set()
    curr_node_name = cli_context.flat_onnx.onnx_graph_node_names_list[0]

    total_nested_graph = rebuild_nested_graph_helper(
        input_to_output_node_dict=input_to_output_node_dict,
        onnx_graph_node_name_to_attributes=cli_context.flat_onnx.onnx_graph_node_name_to_attributes,
        seen_nodes=seen_nodes,
        curr_node_name=curr_node_name,
    )
    with open("total_nested_graph.json", "w") as output_file:
        json.dump(
            total_nested_graph,
            output_file,
            indent=2,
        )


def rebuild_nested_graph_helper(
    input_to_output_node_dict,
    onnx_graph_node_name_to_attributes,
    seen_nodes,
    curr_node_name: str,
):
    if curr_node_name in seen_nodes:
        return

    new_node = dict()
    new_node["name"] = curr_node_name
    new_node["attributes"] = onnx_graph_node_name_to_attributes[curr_node_name]
    seen_nodes.add(curr_node_name)
    new_node["children"] = []

    # Skip if non-existent since we're on the leaf node(s).
    if curr_node_name in input_to_output_node_dict:
        for child_name in input_to_output_node_dict[curr_node_name]:
            new_node["children"].append(
                rebuild_nested_graph_helper(
                    input_to_output_node_dict=input_to_output_node_dict,
                    onnx_graph_node_name_to_attributes=onnx_graph_node_name_to_attributes,
                    seen_nodes=seen_nodes,
                    curr_node_name=child_name,
                )
            )

    return new_node


def generate_outputs(cli_config: CLIConfig, cli_context: CLIContext):
    if not cli_config.enable_output_json:
        click.secho(
            "enable_output_json is disabled, returning early...",
            fg="yellow",
        )
        return

    msg = "enable_output_json is enabled, printing output to {output_directory}...".format(  # noqa: E501
        output_directory=DEFAULT_OUTPUT_JSON_DIRECTORY
    )
    click.secho(
        msg,
        fg="green",
    )

    output_path = Path(DEFAULT_OUTPUT_JSON_DIRECTORY).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    if not isinstance(cli_context.flat_onnx, FlatOnnx):
        msg = "Internal: Expected flat_onnx to be of FlatOnnx type, got {} instead".format(  # noqa: E501
            type(cli_context.flat_onnx)
        )
        logger.error(msg)
        return

    with open(
        output_path.joinpath("onnx_outputs.json"), "w"
    ) as output_file:
        json.dump(
            cli_context.flat_onnx.onnx_outputs,
            output_file,
            indent=cli_config.json_indent_size,
        )

    with open(
        output_path.joinpath("onnx_inputs.json"), "w"
    ) as output_file:
        json.dump(
            cli_context.flat_onnx.onnx_inputs,
            output_file,
            indent=cli_config.json_indent_size,
        )

    with open(
        output_path.joinpath("onnx_graph_node_names_list.json"), "w"
    ) as output_file:
        json.dump(
            cli_context.flat_onnx.onnx_graph_node_names_list,
            output_file,
            indent=cli_config.json_indent_size,
        )


if __name__ == "__main__":
    cli()
