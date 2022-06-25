from collections import defaultdict
import re

from src.converters.graph_code_converter import GraphPythonConverter
from src.converters.utils import to_camel_case, to_snake_case, from_snake_to_normal_str
from utils.algo_utils import GraphAlgos

class ProscriptPythonConverterMethodEdgePredAssert(GraphPythonConverter):

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

                def __init__(self):
                    # steps
                    begin = Step()
                    find_a_notable_theme_park = Step()
                    buy_the_tickets_online = Step()
                    pack_the_bags = Step()
                    head_to_the_car = Step()
                    start_the_car = Step()
                    get_on_the_road = Step()
                    travel_to_the_theme_park = Step()

                    # dependency graph
                    assert begin.before([find_a_notable_theme_park])
                    assert find_a_notable_theme_park.before([buy_the_tickets_online])
                    assert buy_the_tickets_online.before([pack_the_bags])
                    assert pack_the_bags.before([head_to_the_car])
                    assert head_to_the_car.before([start_the_car])
                    assert start_the_car.before([get_on_the_road])
                    assert get_on_the_road.before([travel_to_the_theme_park])
                    assert travel_to_the_theme_park.before([end])

        - If prompt part only is True, then the prompt part of the python code will be returned.
        """
        title = row["scenario"]
        num_steps = len(row["events"])

        class_name = to_camel_case(title)
        class_name = "Plan"  #  re.sub(r'[^a-zA-Z0-9_]', '', class_name)
        py_source = f"class {class_name}:\n\n"""
        py_source += f"    goal = \"{title}\"\n"

        step_to_next_steps_dict = defaultdict(list)
        step_id_to_desc = dict()
        step_names = []
        for step in row["flatten_input_for_edge_prediction"].split("; "):
            step_id, step_description = step.split(": ")
            step_id_to_desc[step_id.strip()] = step_description.strip()
            step_names.append(step_id.strip())
        
        py_source += f"    def __init__(self):\n"
        py_source += f"         # steps\n"
        for step_id in step_names:
            step_description = to_snake_case(step_id_to_desc[step_id])
            py_source += f"        {step_description} = Step()\n"
        py_source += "\n"

        py_source += f"        # dependency graph\n"

        if prompt_part_only:
            return py_source


        for relation in row["flatten_output_for_edge_prediction"].split("; "):
            relation_from, relation_to = relation.split(" -> ")
            step_to_next_steps_dict[relation_from.strip()].append(relation_to.strip())

        step_id_to_next_steps_dict_desc = dict()
        for k, v in step_to_next_steps_dict.items():
            step_id_to_next_steps_dict_desc[step_id_to_desc[k]] = [step_id_to_desc[x] for x in v]

        steps_graph = GraphAlgos.to_nx_graph(step_to_next_steps_dict)
        root_nodes = GraphAlgos.root_nodes(steps_graph)
        assert len(root_nodes) > 0, "There should be at least one root node"
        
        topo_order = GraphAlgos.topo_sort(steps_graph)

        
        for i, step_id in enumerate(topo_order):
            step_desc = to_snake_case(step_id_to_desc[step_id])
            successors = GraphAlgos.get_successors(steps_graph, step_id)
            successors_names = [to_snake_case(step_id_to_desc[x]) for x in successors]
            if len(successors_names) == 0:
                successors_names = ["end"]
            successor_str = ", ".join(successors_names)
            py_source += f"        assert {step_desc}.then([{successor_str}])\n"
            # py_source += f"        {step_desc}.then = [{successor_str}]\n"

        py_source += "\n"

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

                def __init__(self):
                    # steps
                    begin = Step()
                    find_a_notable_theme_park = Step()
                    buy_the_tickets_online = Step()
                    pack_the_bags = Step()
                    head_to_the_car = Step()
                    start_the_car = Step()
                    get_on_the_road = Step()
                    travel_to_the_theme_park = Step()

                    # dependency graph
                    assert begin.before([find_a_notable_theme_park])
                    assert find_a_notable_theme_park.before([buy_the_tickets_online])
                    assert buy_the_tickets_online.before([pack_the_bags])
                    assert pack_the_bags.before([head_to_the_car])
                    assert head_to_the_car.before([start_the_car])
                    assert start_the_car.before([get_on_the_road])
                    assert get_on_the_road.before([travel_to_the_theme_park])
                    assert travel_to_the_theme_park.before([end])

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

        # extract all the steps
        step_names = re.findall(r'(\w+) = Step', py_code_str)
        if step_names[0] == "begin":
            step_names = step_names[1:]

        num_steps = len(step_names)
        step_desc_to_id = {step_names[i]: f"step{i}" for i in range(num_steps)}

        # extract all the dependencies
        relations = []
        dependencies = re.findall(r'assert (\w+).then\(\[(.*)\]\)', py_code_str)
        for dep in dependencies:
            step_name = step_desc_to_id[dep[0].strip()]
            step_deps = []
            for step_dep in dep[1].split(", "):
                step_dep = step_dep.strip()
                if step_dep in step_desc_to_id:
                    step_deps.append(step_desc_to_id[step_dep])
        
            for step_dep in step_deps:
                relations.append(f"{step_name} -> {step_dep}")
        
        result = {
            "title":  title,
            "num_steps": num_steps,
            "schema": [f"{step_desc_to_id[x]}: {from_snake_to_normal_str(x)}" for x in step_names],
            "relations": relations,
        }


        return result
