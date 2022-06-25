from collections import defaultdict
import re

from src.converters.graph_code_converter import GraphPythonConverter
from src.converters.utils import to_camel_case, to_snake_case, from_snake_to_normal_str
from utils.algo_utils import GraphAlgos

class ProscriptPythonConverterMethodExplicit(GraphPythonConverter):

    def graph_to_python(self, row: dict, prompt_part_only: bool) -> str:
        """Converts a proscript schema to a python class.

        Example:
        
        Given:
            Title: travel to the theme park
            Steps: 7
            Schema: step0: step1 description; step1: buy the tickets online; step2: pack the bags; step3: head to the car; step4: start the car; step5: get on the road; step6: travel to the theme park; step0 -> step1; step1 -> step2; step2 -> step3; step3 -> step4; step4 -> step5; step5 -> step6

        Create a code where each step is a method:
            class TravelToThemePark:
                title = "travel to the theme park"
                steps = 7

                def find_a_notable_theme_park():
                    return arg0
 
                def buy_the_tickets_online(arg0):
                    return arg1
                
                def pack_the_bags(arg1):
                    return arg2
                
                def head_to_the_car(arg2):
                    return arg3
                
                def start_the_car(arg3):
                    return arg4
                
                def get_on_the_road(arg4):
                    return arg5
                
                def travel_to_the_theme_park(arg5):
                    return arg6
                
                def main():
                    arg0 = find_a_notable_theme_park()
                    arg1 = buy_the_tickets_online(arg0)
                    arg2 = pack_the_bags(arg1)
                    arg3 = head_to_the_car(arg2)
                    arg4 = start_the_car(arg3)
                    arg5 = get_on_the_road(arg4)
                    arg6 = travel_to_the_theme_park(arg5)
                    return arg6
                

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

        step_to_next_steps_dict = defaultdict(list)
        step_id_to_desc = dict()
        for part in schema.split("; "):
            if "->" in part:
                src, dest = part.split("->")
                step_to_next_steps_dict[src.strip()].append(dest.strip())
            else:
                step_name, step_description = part.split(": ")
                step_id_to_desc[step_name.strip()] = step_description.strip()

        step_id_to_next_steps_dict_desc = dict()
        for k, v in step_to_next_steps_dict.items():
            step_id_to_next_steps_dict_desc[step_id_to_desc[k]] = [step_id_to_desc[x] for x in v]

        steps_graph = GraphAlgos.to_nx_graph(step_to_next_steps_dict)
        root_nodes = GraphAlgos.root_nodes(steps_graph)
        assert len(root_nodes) > 0, "There should be at least one root node"
        
        topo_order = GraphAlgos.topo_sort(steps_graph)
    
        for i, step_name in enumerate(topo_order):
            method_name = to_snake_case(step_id_to_desc[step_name])
            predecessors = GraphAlgos.get_predecessors(steps_graph, step_name)
            arg_list = ", ".join([node.replace("step", "arg") for node in predecessors])
            py_source += f"    def {method_name}({arg_list}):\n"
            py_source += f"        return arg{i}\n\n"

        
        py_source += f"    def main():\n"
        for i, step_name in enumerate(topo_order):
            method_name = to_snake_case(step_id_to_desc[step_name])
            predecessors = GraphAlgos.get_predecessors(steps_graph, step_name)
            arg_list = ", ".join([node.replace("step", "arg") for node in predecessors])
            py_source += f"        arg{i} = {method_name}({arg_list})\n"
            
        py_source += f"        return arg{len(topo_order) - 1}\n"
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

                def find_a_notable_theme_park():
                    return arg0
 
                def buy_the_tickets_online(arg0):
                    return arg1
                
                def pack_the_bags(arg1):
                    return arg2
                
                def head_to_the_car(arg2):
                    return arg3
                
                def start_the_car(arg3):
                    return arg4
                
                def get_on_the_road(arg4):
                    return arg5
                
                def travel_to_the_theme_park(arg5):
                    return arg6
                
                def main():
                    arg0 = find_a_notable_theme_park()
                    arg1 = buy_the_tickets_online(arg0)
                    arg2 = pack_the_bags(arg1)
                    arg3 = head_to_the_car(arg2)
                    arg4 = start_the_car(arg3)
                    arg5 = get_on_the_road(arg4)
                    arg6 = travel_to_the_theme_park(arg5)
                    return arg6
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

        title = re.search(r'goal = "(.*)"', py_code_str).group(1)

        main_func_content = re.search(r'\s*def main\(\):\n(.*)', py_code_str, re.DOTALL).group(1)
        main_func_content.replace("arg", "step")
        main_func_lines = [line.strip() for line in main_func_content.split("\n") if "=" in line]
        schema = []
        for line in main_func_lines:
            step_id, step_description = line.split("=")
            step_description = step_description.split("(")[0]
            step_description = from_snake_to_normal_str(step_description)
            schema.append(f"{step_id.strip()}: {step_description.strip()}")
        
        relations = []
        for line in main_func_lines:
            step_id, step_description = line.split("=")
            step_id = step_id.strip()
            step_parents = step_description.split("(")[1].split(")")[0].split(",")
            step_parents = [x.strip() for x in step_parents if x.strip()]
            for parent in step_parents:
                relations.append(f"{parent} -> {step_id}") 


        result = {
            "title":  title,
            "num_steps": len(schema),
            "schema": schema,
            "relations": relations,
        }


        return result
