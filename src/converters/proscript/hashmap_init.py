from collections import defaultdict
import re

from src.converters.graph_code_converter import GraphPythonConverter
from src.converters.utils import to_camel_case


class ProscriptPythonConverterHashmapInit(GraphPythonConverter):

    def graph_to_python(self, row: dict, prompt_part_only: bool) -> str:
        """Converts a proscript schema to a python class.

        Given:
            Title: travel to the theme park
            Steps: 7
            Schema: step0: find a notable theme park; step1: buy the tickets online; step2: pack the bags; step3: head to the car; step4: start the car; step5: get on the road; step6: travel to the theme park; step0 -> step1; step1 -> step2; step2 -> step3; step3 -> step4; step4 -> step5; step5 -> step6

        Returns:
            class TravelToThemePark:
                title = "travel to the theme park"
                steps = 7

                def __init__(self):
                    step0 = "find a notable theme park"
                    step1 = "buy the tickets online"
                    step2 = "pack the bags"
                    step3 = "head to the car"
                    step4 = "start the car"
                    step5 = "get on the road"
                    step6 = "travel to the theme park"
                    self.nodes = [step0, step1, step2, step3, step4, step5, step6]
                    self.edges = {
                        step0: step1,
                        step1: step2,
                        step2: step3,
                        step3: step4,
                        step4: step5,
                        step5: step6
                    }

        - If prompt part only is True, then the prompt part of the python code will be returned.
        """
        title = row["scenario"]
        num_steps = len(row["events"])
        schema = row["flatten_output_for_script_generation"]
        class_name = to_camel_case(title)
        class_name = re.sub(r'[^a-zA-Z0-9_]', '', class_name)
        py_source = f"class {class_name}:\n\n"""
        py_source += f"    title = \"{title}\"\n"
        py_source += f"    steps = {num_steps}\n\n"

        if prompt_part_only:
            return py_source

        py_source += f"    def __init__(self):\n"

        steps = schema.split("; ")
        relations = []
        nodes = []
        for step in steps:
            if "->" in step:
                relations.append(step)
                continue
            step_name, step_description = step.split(": ")
            step_name = step_name.strip()
            step_description = step_description.strip()
            py_source += f"        {step_name} = \"{step_description}\"\n"
            nodes.append(step_name)
        
        py_source += f"        self.nodes = {nodes}\n"

        # add the edges
        edges = defaultdict(list)
        for relation in relations:
            step_from, step_to = relation.split("->")
            edges[step_from.strip()].append(step_to.strip())


        py_source += "        self.edges = {\n"
        for step_from, step_to_list in edges.items():
            step_str = ", ".join([f"{step_to}" for step_to in step_to_list])
            py_source += f"            {step_from}: [{step_str}],\n"
        py_source += "        }\n"


        return py_source


    def python_to_graph(self, py_code_str: str) -> str:
        """Given a python code string, generates a proscript schema string.

        Example:
            class TravelToThemePark:
                title = "travel to the theme park"
                steps = 7

                def __init__(self):
                    step0 = "find a notable theme park"
                    step1 = "buy the tickets online"
                    step2 = "pack the bags"
                    step3 = "head to the car"
                    step4 = "start the car"
                    step5 = "get on the road"
                    step6 = "travel to the theme park"
                    self.edges = {
                        step0: step1,
                        step1: step2,
                        step2: step3,
                        step3: step4,
                        step4: step5,
                        step5: step6
                    }

        returns:
        [
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
        }
        ]
        """
        # compile the code
        py_code = compile(py_code_str, "<string>", "exec")

        # instantiate the class
        py_code_dict = {}
        exec(py_code, py_code_dict)
        # the newly instantiated class will be last in the scope
        py_code_class = py_code_dict[list(py_code_dict.keys())[1]]()
        result = {
            "title": py_code_class.title,
            "num_steps": py_code_class.steps,
            "schema": [],
            "relations": py_code_class.edges
        }

        for i, step in enumerate(py_code_class.nodes):
            result["schema"] += [f"step{i}: {step}"]

        return result
