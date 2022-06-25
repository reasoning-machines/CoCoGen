from pprint import pprint
import re
from collections import defaultdict

from src.converters.graph_code_converter import GraphPythonConverter
from src.converters.utils import to_camel_case, compile_code_get_object



class ProscriptPythonConverterOO(GraphPythonConverter):


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
                    self.step0 = Step("find a notable theme park")
                    self.step1 = Step("buy the tickets online")
                    self.step2 = Step("pack the bags")
                    self.step3 = Step("head to the car")
                    self.step4 = Step("start the car")
                    self.step5 = Step("get on the road")
                    self.step6 = Step("travel to the theme park")

                def init_edges():
                    self.step0.add_next_steps([self.step1])
                    self.step1.add_next_steps([self.step2])
                    self.step2.add_next_steps([self.step3])
                    self.step3.add_next_steps([self.step4])
                    self.step4.add_next_steps([self.step5])
                    self.step5.add_next_steps([self.step6])

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
        py_source += f"        # make {num_steps} steps\n"
        steps = schema.split("; ")
        relations = []
        for step in steps:
            if "->" in step:
                relations.append(step)
                continue
            step_name, step_description = step.split(": ")
            step_name = step_name.strip()
            step_description = step_description.strip()
            py_source += f"        self.{step_name} = Step(\"{step_description}\")\n"

        # add the edges
        edges = defaultdict(list)
        for relation in relations:
            step_from, step_to = relation.split("->")
            edges[step_from.strip()].append(step_to.strip())
            
        py_source += "\n"
        py_source += f"    def init_edges(self):\n"
        for step_from, step_to_list in edges.items():
            step_str = ", ".join([f"self.{step_to}" for step_to in step_to_list])
            py_source += f"        self.{step_from}.add_next_steps([{step_str}])\n"
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

                def init_steps(self):
                    self.step0 = Step("find a notable theme park")
                    self.step1 = Step("buy the tickets online")
                    self.step2 = Step("pack the bags")
                    self.step3 = Step("head to the car")
                    self.step4 = Step("start the car")
                    self.step5 = Step("get on the road")
                    self.step6 = Step("travel to the theme park")

                def init_edges():
                    self.step0.add_next_steps([self.step1])
                    self.step1.add_next_steps([self.step2])
                    self.step2.add_next_steps([self.step3])
                    self.step3.add_next_steps([self.step4])
                    self.step4.add_next_steps([self.step5])
                    self.step5.add_next_steps([self.step6])
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
            ],
            "relations": [
                "step0 -> step1",
                "step1 -> step2",
                "step2 -> step3",
                "step3 -> step4",
                "step4 -> step5",
                "step5 -> step6"
            ]
        """

        class Step:
            def __init__(self, description):
                self.description = description
                self.next_steps = []
            def add_next_steps(self, next_steps):
                self.next_steps = next_steps

        step_class_str = "class Step:\n"
        step_class_str += "    def __init__(self, description):\n"
        step_class_str += "        self.description = description\n"
        step_class_str += "        self.next_steps = []\n"
        step_class_str += "    def add_next_steps(self, next_steps):\n"
        step_class_str += "        self.next_steps = next_steps\n"


        py_code_class = compile_code_get_object(f"{step_class_str}\n" + py_code_str)

        py_code_class.init_steps()
        py_code_class.init_edges()

        # src = py_code_str.split("\n")
        # for i, line in enumerate(src):
        #     # show number with 2 digits
        #     print(f"{(i + 1):02d}: {line}")



        result = {
            "title": py_code_class.title,
            "num_steps": py_code_class.steps,
            "schema": [],
            "relations": [],
        }

        step_to_step_id = dict()
        
        nodes = []

        # just in case the graph has generated more steps, capture those as well 
        # instead of crashing

        for i in range(0 ,100):
            try:
                step_id = f"step{i}"
                step = getattr(py_code_class, step_id)
                nodes.append(step)
                result["schema"] += [f"{step_id}: {step.description}"]
                step_to_step_id[step.description] = step_id
            except AttributeError:  # will break whenever it hits a step that doesn't exist
                break


        relations = []

        
        for node in nodes:
            node_from_id = step_to_step_id[node.description]
            for next_step in node.next_steps:
                node_to_id = step_to_step_id[next_step.description]
                relations.append(f"{node_from_id} -> {node_to_id}")

        result["relations"] = relations
            

        return result
