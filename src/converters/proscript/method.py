from collections import defaultdict
import re

from src.converters.graph_code_converter import GraphPythonConverter
from src.converters.utils import to_camel_case, to_snake_case, from_snake_to_normal_str
from utils.algo_utils import GraphAlgos

class ProscriptPythonConverterMethod(GraphPythonConverter):

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

                def begin():
                    find_a_notable_theme_park()

                def find_a_notable_theme_park():
                    buy_the_tickets_online()
                
                def buy_the_tickets_online():
                    pack_the_bags()
                
                def pack_the_bags():
                    head_to_the_car()
                
                def head_to_the_car():
                    start_the_car()
                
                def start_the_car():
                    get_on_the_road()
                
                def get_on_the_road():
                    travel_to_the_theme_park()
                
                def travel_to_the_theme_park():
                    end()
                

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

        src_to_dest_dict = defaultdict(list)
        node_id_to_desc = dict()
        for part in schema.split("; "):
            if "->" in part:
                src, dest = part.split("->")
                src_to_dest_dict[src.strip()].append(dest.strip())
            else:
                step_name, step_description = part.split(": ")
                node_id_to_desc[step_name.strip()] = step_description.strip()

        src_to_dest_dict_desc = dict()
        for k, v in src_to_dest_dict.items():
            src_to_dest_dict_desc[node_id_to_desc[k]] = [node_id_to_desc[x] for x in v]

        src_to_dest_dict = src_to_dest_dict_desc
        src_dest_nx_graph = GraphAlgos.to_nx_graph(src_to_dest_dict_desc)
        root_nodes = GraphAlgos.root_nodes(src_dest_nx_graph)
        assert len(root_nodes) > 0, "There should be at least one root node"
        
        topo_order = GraphAlgos.topo_sort(src_dest_nx_graph)
        
        py_source += f"    def begin():\n"
        
        for root_node in root_nodes:
            py_source += f"        {to_snake_case(root_node)}()\n"
        py_source += f"\n"
    
        for step_name in topo_order:
            method_name = to_snake_case(step_name)
            py_source += f"    def {method_name}():\n"
            
            if step_name not in src_to_dest_dict:
                py_source += f"        end()\n"
            else:
                for nbr in src_to_dest_dict[step_name]:
                    nbr_method_name = to_snake_case(nbr)
                    py_source += f"        {nbr_method_name}()\n"
            py_source += f"\n"
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
            goal = "travel to the theme park"
            steps = 7

            def begin():
                find_a_notable_theme_park()

            def find_a_notable_theme_park():
                buy_the_tickets_online()
            
            def buy_the_tickets_online():
                pack_the_bags()
            
            def pack_the_bags():
                head_to_the_car()
            
            def head_to_the_car():
                start_the_car()
            
            def start_the_car():
                get_on_the_road()
            
            def get_on_the_road():
                travel_to_the_theme_park()
            
            def travel_to_the_theme_park():
                end()

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

        
        ## This is pretty much hand-built messy parsing.
        ### Iterate over each line, if there's a method call, add it to the dict, maintain state

        method_to_calls_dict = defaultdict(list)
        py_code_lines = py_code_str.split("\n")

        path_method_def = r'\s*def (.*)\('
        
        i = 0
        
        inside_function_definition = False
        is_function_definition = False
        
        while i < len(py_code_lines):

            curr_line = py_code_lines[i].strip()
            
            is_function_definition = re.match(path_method_def, curr_line) is not None

            if is_function_definition:
                
                curr_method_name = from_snake_to_normal_str(re.search(path_method_def, curr_line).group(1))
                method_to_calls_dict[curr_method_name] = []
                i += 1
                inside_function_definition = True  # inside a function definition now

                while i < len(py_code_lines) and inside_function_definition:

                    curr_line = py_code_lines[i].strip()
                    if len(curr_line) == 0:
                        i += 1
                        continue
                    # our functions are simple, we can just look for calls to other methods except for empty lines
                    inside_function_definition = re.match(path_method_def, curr_line) is None
                    
                    if inside_function_definition:
                        
                        call_method_name = from_snake_to_normal_str(re.search(r'(\w+)\(', curr_line).group(1))
                        method_to_calls_dict[curr_method_name].append(call_method_name)
                        i += 1

            else:
                i += 1

        # method_to_calls_dict is a graph: the key is the source, and value is a list of destinations
        method_to_calls_graph = GraphAlgos.to_nx_graph(method_to_calls_dict)
        topo_order = GraphAlgos.topo_sort(method_to_calls_graph)
        
        # the generated code may not always have begin and end
        if topo_order[0] == "begin":
            topo_order = topo_order[1:]
        if topo_order[-1] == "end":
            topo_order = topo_order[:-1]
        
        node_to_node_id = {node: f"step{i}" for i, node in enumerate(topo_order)}

        relations = []
        for node, nbrs in method_to_calls_dict.items():
            if node == "begin":
                continue
            for nbr in nbrs:
                if nbr == "end":
                    continue
                node_id = node_to_node_id[node]
                nbr_id = node_to_node_id[nbr]
                relations.append(f"{node_id} -> {nbr_id}")

        result = {
            "title":  title,
            "num_steps": len(node_to_node_id),
            "schema": [f"step{i}: {node}" for i, node in enumerate(topo_order)],
            "relations": relations,
        }


        return result
