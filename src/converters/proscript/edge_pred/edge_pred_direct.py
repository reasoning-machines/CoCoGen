import re

from collections import defaultdict
import re

from src.converters.graph_code_converter import GraphPythonConverter
from src.converters.utils import to_camel_case, to_snake_case, from_snake_to_normal_str
from utils.algo_utils import GraphAlgos



class ProscriptPythonConverterEdgePredDirect(GraphPythonConverter):

    def graph_to_python(self, row: dict, prompt_part_only: bool) -> str:
        """Converts a proscript schema to a python class.

        Example:
            Proscript:
            Title: travel to the theme park
            Steps: 7
            Schema: step0: find a notable theme park; step1: buy the tickets online; step2: pack the bags; step3: head to the car; step4: start the car; step5: get on the road; step6: travel to the theme park; step0 -> step1; step1 -> step2; step2 -> step3; step3 -> step4; step4 -> step5; step5 -> step6

            class TravelToThemePark:
                title = "travel to the theme park"
                steps = 7
                def __init__(self):
                    self.nodes = "find a notable theme park; buy the tickets online; pack the bags; head to the car; start the car; get on the road; travel to the theme park"
                    self.edges = "step0 -> step1; step1 -> step2; step2 -> step3; step3 -> step4; step4 -> step5; step5 -> step6"

        - If prompt part only is True, then the prompt part of the python code will be returned.
        """
        title = row["scenario"]
        num_steps = len(row["events"])

        class_name = to_camel_case(title)
        class_name = re.sub(r'[^a-zA-Z0-9_]', '', class_name)
        py_source = f"class {class_name}:\n\n"""
        py_source += f"    goal = \"{title}\"\n"
        py_source += f"    steps = {num_steps}\n\n"

        py_source += f"    def __init__(self):\n"
        py_source += f"        self.nodes = \"{row['flatten_input_for_edge_prediction']}\"\n\n"
        
        if prompt_part_only:
            return py_source
        
        py_source += f"        self.edges = \"{row['flatten_output_for_edge_prediction']}\"\n\n"

        return py_source


    def python_to_graph(self, py_code_str: str) -> str:
        # calls the converter, if there's an exception, reduce one line and try again

        lines_to_reduce = [None] + list(range(1, 3))
        for line_to_reduce in lines_to_reduce:
            try:
                if line_to_reduce is not None:
                    py_code_lines = py_code_str.split("\n")
                    py_code_lines_updated = "\n".join(py_code_lines[:-line_to_reduce])
                else:
                    py_code_lines_updated = py_code_str
                return self._python_to_graph(py_code_lines_updated)

            except Exception as e:
                print(e)
                print("Exception occurred, trying again with line reduction")
                continue


    def _python_to_graph(self, py_code_str: str) -> str:
        """Given a python code string, generates a proscript schema string.

        Example:
            class TravelToThemePark:
                title = "travel to the theme park"
                steps = 7
                def __init__(self):
                    self.nodes = "find a notable theme park; buy the tickets online; pack the bags; head to the car; start the car; get on the road; travel to the theme park"
                    self.edges = "step0 -> step1; step1 -> step2; step2 -> step3; step3 -> step4; step4 -> step5; step5 -> step6"

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
        
        print(py_code_str)
        title = re.search(r'goal = "(.*)"', py_code_str).group(1)

        # extract the steps
        steps = re.search(r'nodes = "(.*)"', py_code_str).group(1)
        steps = steps.split("; ")
        steps = [step.strip() for step in steps]

        # extract the edges
        edges = re.search(r'edges = "(.*)"', py_code_str).group(1)
        edges = edges.split("; ")
        edges = [edge.strip() for edge in edges]

        
        result = {
            "title":  title,
            "num_steps": len(steps),
            "schema": steps,
            "relations": edges,
        }

        return result
