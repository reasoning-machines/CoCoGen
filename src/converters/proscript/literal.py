import re

from src.converters.graph_code_converter import GraphPythonConverter
from src.converters.utils import to_camel_case, compile_code_get_object


class ProscriptPythonConverterLiteral(GraphPythonConverter):

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
        schema = row["flatten_output_for_script_generation"]
        class_name = to_camel_case(title)
        class_name = re.sub(r'[^a-zA-Z0-9_]', '', class_name)
        py_source = f"class {class_name}:\n\n"""
        py_source += f"    title = \"{title}\"\n"
        py_source += f"    steps = {num_steps}\n\n"

        if prompt_part_only:
            return py_source

        steps = schema.split("; ")
        relations = []
        for step in steps:
            if "->" in step:
                relations.append(step)
                continue
            step_name, step_description = step.split(": ")
            step_name = step_name.strip()
            step_description = step_description.strip()
            py_source += f"    def {step_name}(self):\n"
            py_source += f"        return \"{step_description}\"\n\n"

        # add the relations
        py_source += "    def get_relations(self):\n"
        py_source += "        return [\n"
        for relation in relations:
            relation_from, relation_to = relation.split(" -> ")
            py_source += f"            \"{relation_from} -> {relation_to}\",\n"
        py_source += "        ]\n"

        return py_source

    def python_to_graph(self, py_code_str: str) -> str:
        """Given a python code string, generates a proscript schema string.

        Example:
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
            ]
            }
        """
        # compile the code
        py_code_class = compile_code_get_object(py_code_str)

        result = {
            "title": py_code_class.title,
            "num_steps": py_code_class.steps,
            "schema": [],
            "relations": py_code_class.get_relations(),
        }

        num_steps = py_code_class.steps
        for i in range(num_steps):
            step_func = f"step{i}"
            step_description = getattr(py_code_class, step_func)()
            result["schema"] += [f"step{i}: {step_description}"]

        return result
