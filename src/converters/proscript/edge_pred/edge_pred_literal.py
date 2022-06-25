import re

from src.converters.proscript.literal import ProscriptPythonConverterLiteral
from src.converters.utils import to_camel_case, compile_code_get_object


class ProscriptPythonConverterEdgePredLiteral(ProscriptPythonConverterLiteral):

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
                    pass

                def step0(self):
                    return "find a notable theme park"

                def step1(self):
                    return "buy the tickets online"

                def step2(self):
                    return "pack the bags"

                def step3(self):
                    return "head to the car"

                def step4(self):
                    return "start the car"

                def step5(self):
                    return "get on the road"

                def step6(self):
                    return "travel to the theme park"

        - If prompt part only is True, then the prompt part of the python code will be returned.
        """
        title = row["scenario"]
        num_steps = len(row["events"])

        class_name = to_camel_case(title)
        class_name = re.sub(r'[^a-zA-Z0-9_]', '', class_name)
        py_source = f"class {class_name}:\n\n"""
        py_source += f"    title = \"{title}\"\n"
        py_source += f"    steps = {num_steps}\n\n"

        
        for step in row["flatten_input_for_edge_prediction"].split("; "):
            step_name, step_description = step.split(": ")
            py_source += f"    def {step_name.strip()}(self):\n"
            py_source += f"        return \"{step_description.strip()}\"\n\n"

        if prompt_part_only:
            return py_source

        # add the relations
        py_source += "    def get_relations(self):\n"
        py_source += "        return [\n"
        for relation in row["flatten_output_for_edge_prediction"].split("; "):
            relation_from, relation_to = relation.split(" -> ")
            py_source += f"            \"{relation_from.strip()} -> {relation_to.strip()}\",\n"
        py_source += "        ]\n"

        return py_source

