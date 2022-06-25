import re

from src.converters.proscript.networkx import ProscriptPythonConverterNetworkx
from src.converters.utils import to_camel_case


class ProscriptPythonConverterEdgePredNetworkx(ProscriptPythonConverterNetworkx):

    def graph_to_python(self, row: dict, prompt_part_only: bool) -> str:
        """Converts a proscript schema to a python class for edge prediction.
        Please see the base class for detailed documentation.
        The python_to_graph function is used from the base class.

        - If prompt part only is True, then the prompt part of the python code will be returned. In this case, that is all the steps.
        """
        title = row["scenario"]
        num_steps = len(row["events"])

        class_name = to_camel_case(title)
        class_name = "Plan"  #  re.sub(r'[^a-zA-Z0-9_]', '', class_name)
        py_source = f"class {class_name}:\n\n"""
        py_source += f"    goal = \"{title}\"\n"
        py_source += f"    num_steps = {num_steps}\n\n"


        py_source += f"    def __init__(self):\n"
        py_source += f"        graph = nx.DiGraph()\n"


        step_names = []

        py_source += f"        # add nodes\n"
        for step in row["flatten_input_for_edge_prediction"].split("; "):
            step_name, step_description = step.split(": ")
            step_name = step_name.strip()
            step_names.append(step_name)
            py_source += f"        {step_name} = \"{step_description.strip()}\"\n"

        py_source += f"        graph.add_nodes_from([{', '.join(step_names)}])\n\n"


        if prompt_part_only:
            return py_source

        py_source += "        # add edges\n"
        # add the relations
        for relation in row["flatten_output_for_edge_prediction"].split("; "):
            relation_from, relation_to = relation.split(" -> ")
            py_source += f"        graph.add_edge({relation_from.strip()}, {relation_to.strip()})\n"

        return py_source

