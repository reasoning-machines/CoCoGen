import re

from src.converters.graph_code_converter import GraphPythonConverter
from src.converters.utils import from_snake_to_normal_str

class ProscriptPythonConverterMethodEdgePredDepFixer(GraphPythonConverter):

    def get_diff(self, ref_dep_dict: dict, gen_dep_dict: dict) -> str:
        fixes = []
        misses = []
        for src, ref_deps in ref_dep_dict.items():
            if src not in gen_dep_dict:
                misses.append(f"{src}.requires = [{ref_deps}]")
            elif gen_dep_dict[src] != ref_deps:
                misses.append(f"{src}.requires = [{ref_deps}]")
                fixes.append(f"if {src}.requires != {ref_deps}:")
                fixes.append(f"    {src}.requires = [{ref_deps}]")
        return misses, fixes
                

    def graph_to_python(self, row: dict, prompt_part_only: bool) -> str:
        """Generates an edit between reference code and generated code.
       - If prompt part only is True, then the prompt part of the python code will be returned (the bad code).
        """ 
        generated_code = row["generated_code"]
        reference_code = row["reference_code"]



        if prompt_part_only:
            return generated_code

        generated_dependencies = re.findall(r'(\w+)\.requires = \[(.*)\]', generated_code)
        reference_dependencies = re.findall(r'(\w+)\.requires = \[(.*)\]', reference_code)

        generated_dependencies_dict = {k: v for k, v in generated_dependencies}
        reference_dependencies_dict = {k: v for k, v in reference_dependencies}
        misses, _ = self.get_diff(ref_dep_dict=reference_dependencies_dict, gen_dep_dict=generated_dependencies_dict)
        
        generated_code += "        def fixed_dependencies(self):\n"
        if len(misses) == 0:
            generated_code += "            return []\n"
        else:
            generated_code += "            return [\n"
            for miss in misses[:-1]:
                generated_code += f"                {miss},\n"
            generated_code += f"                {misses[-1]}\n"
            generated_code += "            ]\n\n"

        return generated_code



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
                print(py_code_str)
                raise e
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
                    find_a_notable_theme_park.requires = [begin]
                    buy_the_tickets_online.requires = [find_a_notable_theme_park]
                    pack_the_bags.requires = [buy_the_tickets_online]
                    head_to_the_car.requires = [pack_the_bags]
                    start_the_car.requires = [head_to_the_car]
                    get_on_the_road.requires = [start_the_car]
                    travel_to_the_theme_park.requires = [get_on_the_road]
                
                def fixed_dependencies(self):
                    return [
                        find_a_notable_theme_park.requires = [begin]
                        buy_the_tickets_online.requires = [find_a_notable_theme_park]
                        pack_the_bags.requires = [buy_the_tickets_online]
                    ]
        1. Extracts all the dependencies from the fixed_dependencies method.
        2. Replaces the dependencies in the fixed_dependencies method with the dependencies extracted from the graph.


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
        
        where the relations are a union of original and fixed dependencies.
        """
        
        title = re.search(r'goal = "(.*)"', py_code_str).group(1)

        # extract all the steps
        step_names = re.findall(r'(\w+) = Step', py_code_str)
        if step_names[0] == "begin":
            step_names = step_names[1:]

        num_steps = len(step_names)
        step_desc_to_id = {step_names[i]: f"step{i}" for i in range(num_steps)}

        # extract all the original dependencies
        code_before_fixed_dependencies = py_code_str.split("def fixed_dependencies(self):")[0]
        generated_dependencies = re.findall(r'(\w+)\.requires = \[(.*)\]', code_before_fixed_dependencies)
        
        code_after_fixed_dependencies = py_code_str.split("def fixed_dependencies(self):")[1]
        fixed_dependencies = re.findall(r'(\w+)\.requires = \[(.*)\]', code_after_fixed_dependencies)
        fixed_dependencies_dict = {dep[0]: dep[1] for dep in fixed_dependencies}

        relations = []

        for (source, dependencies) in generated_dependencies:
            if source.strip() not in step_desc_to_id:
                continue
            if source in fixed_dependencies_dict:
                dependencies = fixed_dependencies_dict[source]
            
            parsed_step_deps = []

            
            for step_dep in dependencies.split(", "):
                step_dep = step_dep.strip()
                if step_dep in step_desc_to_id:
                    parsed_step_deps.append(step_desc_to_id[step_dep])
            
            step_name = step_desc_to_id[source.strip()]
            for step_dep in parsed_step_deps:
                relations.append(f"{step_dep} -> {step_name}")
        
        result = {
            "title":  title,
            "num_steps": num_steps,
            "schema": [f"{step_desc_to_id[x]}: {from_snake_to_normal_str(x)}" for x in step_names],
            "relations": relations,
        }


        return result


def unit_test():
    reference_code = """
            class UnitTest:
                goal = "unit test"
                steps = 4

                def __init__(self):
                    # steps
                    begin = Step()
                    a = Step()
                    b = Step()
                    c = Step()
                    d = Step()

                    # dependency graph
                    a.requires = [begin]
                    b.requires = [begin]
                    c.requires = [a, b]
                    d.requires = [c]
    """

    generated_code = """
            class UnitTest:
                goal = "unit test"
                steps = 4

                def __init__(self):
                    # steps
                    begin = Step()
                    a = Step()
                    b = Step()
                    c = Step()
                    d = Step()

                    # dependency graph
                    a.requires = [begin]
                    b.requires = [a]
                    c.requires = [b]
                    d.requires = [c]

    """

    code_with_fix = """
            class UnitTest:
                goal = "unit test"
                steps = 4

                def __init__(self):
                    # steps
                    begin = Step()
                    a = Step()
                    b = Step()
                    c = Step()
                    d = Step()

                    # dependency graph
                    a.requires = [begin]
                    b.requires = [a]
                    c.requires = [b]
                    d.requires = [c]

                def fixed_dependencies(self):
                    return [
                        b.requires = [begin],
                        c.requires = [a, b]
                    ]

    """

    generated_graph = {
        "title": "unit test",
        "num_steps": 4,
        "schema": [
            "step0: a",
            "step1: b",
            "step2: c",
            "step3: d",
        ],
        "relations": [
            "step0 -> step1",
            "step1 -> step2",
            "step2 -> step3",
        ]

    }

    fixed_graph = {
        "title": "unit test",
        "num_steps": 4,
        "schema": [
            "step0: a",
            "step1: b",
            "step2: c",
            "step3: d",
        ],
        "relations": [
            "step0 -> step2",
            "step1 -> step2",
            "step2 -> step3",
        ]

    }
    row = dict()
    row["reference_code"] = reference_code
    row["generated_code"] = generated_code

    converter = ProscriptPythonConverterMethodEdgePredDepFixer()
    result = converter.graph_to_python(row=row, prompt_part_only=False)
    result_prompt_only = converter.graph_to_python(row=row, prompt_part_only=True)
    # print(result)
    assert result.strip() == code_with_fix.strip(), f"{result} != {code_with_fix}"
    # result_parts = result.strip().split("\n")
    # code_with_fix_parts = code_with_fix.strip().split("\n")
    # print(len(result_parts), len(code_with_fix_parts))
    # for i in range(len(result_parts)):
    #     if result_parts[i] != code_with_fix_parts[i]:
    #         print(f"{i}: {result_parts[i]} != {code_with_fix_parts[i]}")
    #     else:
    #         print(f"{i}: {result_parts[i]} == {code_with_fix_parts[i]}")

    result = converter.python_to_graph(py_code_str=result)
    
    fixed_relations = set(result["relations"])
    reference_relations = set(fixed_graph["relations"])
    generated_relations = set(generated_graph["relations"])
    assert fixed_relations == reference_relations, f"{fixed_relations} != {reference_relations}"
    assert generated_relations != reference_relations, f"{generated_relations} != {reference_relations}"


if __name__ == "__main__":
    unit_test()