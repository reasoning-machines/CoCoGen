from collections import defaultdict
from pprint import pprint
import re

from src.converters.graph_code_converter import GraphPythonConverter
from src.converters.utils import to_camel_case, compile_code_get_object


class ProscriptPythonConverterHashmapSep(GraphPythonConverter):

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

                def init_steps(self):
                    self.step0 = "find a notable theme park"
                    self.step1 = "buy the tickets online"
                    self.step2 = "pack the bags"
                    self.step3 = "head to the car"
                    self.step4 = "start the car"
                    self.step5 = "get on the road"
                    self.step6 = "travel to the theme park"

                def init_edges():
                    self.edges = {
                        self.step0: [self.step1],
                        self.step1: [self.step2],
                        self.step2: [self.step3],
                        self.step3: [self.step4],
                        self.step4: [self.step5],
                        self.step5: [self.step6]
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

        py_source += f"    def init_steps(self):\n"

        steps = schema.split("; ")
        relations = []
        for step in steps:
            if "->" in step:
                relations.append(step)
                continue
            step_name, step_description = step.split(": ")
            step_name = step_name.strip()
            step_description = step_description.strip()
            py_source += f"        self.{step_name} = \"{step_description}\"\n"

        # add the edges
        edges = defaultdict(list)
        for relation in relations:
            step_from, step_to = relation.split("->")
            edges[step_from.strip()].append(step_to.strip())
            
        py_source += "\n"
        py_source += f"    def init_edges(self):\n"
        py_source += "        self.edges = {\n"
        for step_from, step_to_list in edges.items():
            step_str = ", ".join([f"self.{step_to}" for step_to in step_to_list])
            py_source += f"            self.{step_from}: [{step_str}],\n"
        py_source += "        }\n"
        return py_source


    def python_to_graph(self, py_code_str: str) -> str:
        """Given a python code string, generates a proscript schema string.

        Example:
            class TravelToThemePark:
                title = "travel to the theme park"
                steps = 7

                def init_steps(self):
                    self.step0 = "find a notable theme park"
                    self.step1 = "buy the tickets online"
                    self.step2 = "pack the bags"
                    self.step3 = "head to the car"
                    self.step4 = "start the car"
                    self.step5 = "get on the road"
                    self.step6 = "travel to the theme park"

                def init_edges():
                    self.edges = {
                        self.step0: self.step1,
                        self.step1: self.step2,
                        self.step2: self.step3,
                        self.step3: self.step4,
                        self.step4: self.step5,
                        self.step5: self.step6
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
        py_code_class = compile_code_get_object(py_code_str)


        py_code_class.init_steps()
        py_code_class.init_edges()
        result = {
            "title": py_code_class.title,
            "num_steps": py_code_class.steps,
            "schema": [],
            "relations": None
        }

        num_steps = py_code_class.steps
        step_to_step_id = dict()
        for i in range(num_steps):
            step_func = f"step{i}"
            step_description = getattr(py_code_class, step_func)
            result["schema"] += [f"step{i}: {step_description}"]
            step_to_step_id[step_description] = step_func

        # just in case the graph has generated more steps, capture those as well 
        # instead of crashing
        for i in range(num_steps, 100):
            try:
                step_func = f"step{i}"
                step_description = getattr(py_code_class, step_func)
                result["schema"] += [f"step{i}: {step_description}"]
                step_to_step_id[step_description] = step_func
            except AttributeError:  # will break whenever it hits a step that doesn't exist
                break

        relations = []

        for step_from, step_to_list in py_code_class.edges.items():
            step_from_id = step_to_step_id[step_from]
            step_to_id_list = [step_to_step_id[step_to] for step_to in step_to_list]
            for step_to_id in step_to_id_list:
                relations.append(f"{step_from_id} -> {step_to_id}")
        
        result["relations"] = relations
            

        return result
