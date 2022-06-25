# simple text converter for testing curie

import re

from src.converters.graph_code_converter import GraphPythonConverter


class ProscriptPythonConverterEdgePredText(GraphPythonConverter):

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

        input = row["flatten_input_for_edge_prediction"]
        schema = row["flatten_output_for_edge_prediction"]

        output = f"Input = {input}\n\n"

        if prompt_part_only:
            return output

        output += f"Output = {schema}\n\n"

        return output

    def python_to_graph(self, py_code_str: str) -> str:
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

        nodes = re.search(r"Input = (.*)", py_code_str).group(1).split("; ")
        output = py_code_str.split("Output =")[1]
        relations = [r.strip() for r in output.split("; ")]

        result = {
            "title": "N/A",
            "num_steps": len(nodes),
            "schema": nodes,
            "relations": relations
        }

        return result
