import re

from src.converters.graph_code_converter import GraphPythonConverter
from src.converters.utils import to_camel_case


class ProscriptPythonConverterNetworkx(GraphPythonConverter):

    def graph_to_python(self, row: dict, prompt_part_only: bool) -> str:
        """Converts a proscript schema to a python class.

        Example:
        
        Given:
            Title: travel to the theme park
            Steps: 7
            Schema: step0: find a notable theme park; step1: buy the tickets online; step2: pack the bags; step3: head to the car; step4: start the car; step5: get on the road; step6: travel to the theme park; step0 -> step1; step1 -> step2; step2 -> step3; step3 -> step4; step4 -> step5; step5 -> step6

        Create a networkx graph:

            class TravelToThemePark:
                title = "travel to the theme park"
                steps = 7

                def __init__(self):
                    graph = nx.DiGraph()
                    
                    # add nodes
                    step0 = graph.add_node("find a notable theme park")
                    step1 = graph.add_node("buy the tickets online")
                    step2 = graph.add_node("pack the bags")
                    step3 = graph.add_node("head to the car")
                    step4 = graph.add_node("start the car")
                    step5 = graph.add_node("get on the road")
                    step6 = graph.add_node("travel to the theme park")

                    # add edges
                    graph.add_edge(step0, step1)
                    graph.add_edge(step1, step2)
                    graph.add_edge(step2, step3)
                    graph.add_edge(step3, step4)
                    graph.add_edge(step4, step5)
                    graph.add_edge(step5, step6)

        - If prompt part only is True, then the prompt part of the python code will be returned.
        """
        title = row["scenario"]
        num_steps = len(row["events"])
        schema = row["flatten_output_for_script_generation"]
        class_name = to_camel_case(title)
        class_name = "Plan"  #  re.sub(r'[^a-zA-Z0-9_]', '', class_name)
        py_source = f"class {class_name}:\n\n"""
        py_source += f"    goal = \"{title}\"\n"
        py_source += f"    num_steps = {num_steps}\n\n"

        if prompt_part_only:
            return py_source

        py_source += f"    def __init__(self):\n"
        py_source += f"        graph = nx.DiGraph()\n"

        steps = schema.split("; ")
        relations = []

        step_names = []

        py_source += f"        # add nodes\n"
        for step in steps:
            if "->" in step:
                relations.append(step)
                continue
            step_name, step_description = step.split(": ")
            step_name = step_name.strip()
            step_description = step_description.strip()
            step_names.append(step_name)
            py_source += f"        {step_name} = \"{step_description}\"\n"

        py_source += f"        graph.add_nodes_from([{', '.join(step_names)}])\n\n"

        py_source += "        # add edges\n"
        # add the relations
        for relation in relations:
            relation_from, relation_to = relation.split(" -> ")
            relation_from = relation_from.strip()
            relation_to = relation_to.strip()
            py_source += f"        graph.add_edge({relation_from}, {relation_to})\n"

        return py_source

    def python_to_graph(self, py_code_str: str) -> str:
        """Given a python code string, generates a proscript schema string.

        Example:
            class TravelToThemePark:
                title = "travel to the theme park"
                steps = 7

                def __init__(self):
                    graph = nx.DiGraph()
                    
                    # add nodes
                    step0 = graph.add_node("find a notable theme park")
                    step1 = graph.add_node("buy the tickets online")
                    step2 = graph.add_node("pack the bags")
                    step3 = graph.add_node("head to the car")
                    step4 = graph.add_node("start the car")
                    step5 = graph.add_node("get on the road")
                    step6 = graph.add_node("travel to the theme park")

                    # add edges
                    graph.add_edge(step0, step1)
                    graph.add_edge(step1, step2)
                    graph.add_edge(step2, step3)
                    graph.add_edge(step3, step4)
                    graph.add_edge(step4, step5)
                    graph.add_edge(step5, step6)

        returns:
            {
            "title": "travel to the theme park",
            "num_steps": 7,
                "schema": [
                        "step0: find a notable theme park",
                        "step1: buy the tickets online",
                        "step2: pack the bags",
                        "step3: head to the car",
                        "step4: start the car",
                        "step5: get on the road",
                        "step6: travel to the theme park",
                ],
                "relations": [
                    "step0 -> step1",
                    "step1 -> step2",
                    "step2 -> step3",
                    "step3 -> step4",
                    "step4 -> step5",
                    "step5 -> step6"
                ]
            }
        """

        code_lines = py_code_str.split("\n")

        # node creation lines
        node_creation_lines = [line for line in code_lines if "=" in line and "step" in line and "num_steps" not in line]

        # extract nodes
        nodes = []
        for node_creation_line in node_creation_lines:
            # get step number
            step_number = node_creation_line.split("=")[0].strip()
            step_description = node_creation_line.split("=")[1].strip().replace("\"", "")
            nodes.append(f"{step_number}: {step_description}")

        # extract edges
        edges = []
        for line in code_lines:
            if "add_edge" in line:
                edge_from = line.split("add_edge(")[1].split(",")[0].strip()
                edge_to = line.split("add_edge(")[1].split(",")[1].split(")")[0].strip()
                edges.append(f"{edge_from} -> {edge_to}")


        for line in code_lines:
            if "goal" in line:
                title = line.split("=")[1].strip().replace("\"", "")

        result = {
            "title":  title,
            "num_steps": len(nodes),
            "schema": nodes,
            "relations": edges,
        }


        return result
