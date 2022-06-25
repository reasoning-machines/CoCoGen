# simple text converter for testing curie

from collections import defaultdict
import re

from src.converters.graph_code_converter import GraphPythonConverter
from src.converters.utils import to_camel_case


class ProscriptPythonConverterText(GraphPythonConverter):

    def graph_to_python(self, row: dict, prompt_part_only: bool) -> str:
        """Converts a proscript schema to a simple text prompt.

        Given:
            Title: travel to the theme park
            Steps: 7
            Schema: step0: find a notable theme park; step1: buy the tickets online; step2: pack the bags; step3: head to the car; step4: start the car; step5: get on the road; step6: travel to the theme park; step0 -> step1; step1 -> step2; step2 -> step3; step3 -> step4; step4 -> step5; step5 -> step6

        Returns:
        Input: You want to travel to the theme park. How can you do this in 7 steps?
        Output: step0: find a notable theme park; step1: buy the tickets online; step2: pack the bags; step3: head to the car; step4: start the car; step5: get on the road; step6: travel to the theme park; step0 -> step1; step1 -> step2; step2 -> step3; step3 -> step4; step4 -> step5; step5 -> step6
        - If prompt part only is True, then the prompt part of the python code will be returned.
        """

        input = row["flatten_input_for_script_generation"]
        schema = row["flatten_output_for_script_generation"]

        output = f"Input = {input}\n\n"

        if prompt_part_only:
            return output

        output += f"Output = {schema}\n\n"

        return output

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
        """Given
        Input: You want to travel to the theme park. How can you do this in 7 steps?
        Output: step0: find a notable theme park; step1: buy the tickets online; step2: pack the bags; step3: head to the car; step4: start the car; step5: get on the road; step6: travel to the theme park; step0 -> step1; step1 -> step2; step2 -> step3; step3 -> step4; step4 -> step5; step5 -> step6

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

        try:
            title = re.search(
                r'Input = (.*). How can you do this in (\d+) steps\?', py_code_str).group(1)
        except:
            title = "N/A"

        try:
            num_steps = re.search(
                r'Input = (.*). How can you do this in (\d+) steps\?', py_code_str).group(2)
        except:
            num_steps = -1

        schema = py_code_str.split("Output =")[1]
        parts = schema.split("; ")
        nodes = []
        relations = []
        for part in parts:
            part = part.strip()
            if "->" in part:
                relations.append(part)
            else:
                nodes.append(part)

        result = {
            "title": title,
            "num_steps": num_steps,
            "schema": nodes,
            "relations": relations
        }

        return result
