import re
from src.converters.graph_code_converter import GraphPythonConverter



class ExplagraphPythonConverterLiteral(GraphPythonConverter):

    def graph_to_python(self, row: dict, prompt_part_only: bool) -> str:
        """Converts a proscript schema to a python class.

        Example:
            {
                "belief": "Marijuana should not be legalized.",
                "argument": "Marijuana is dangerous for society.",
                "stance": "support",
                "graph": "(marijuana; is a; recreational drug)(recreational drug; capable of; drug addiction)(drug addiction; is a; dangerous for society)(dangerous for society; not desires; legalized)",
                "question": "Belief: Marijuana should not be legalized., argument: Marijuana is dangerous for society.",
                "answer": "stance = support | digraph G {\n  \"marijuana\" -> \"recreational drug\" [label=\"is a\"];\n  \"recreational drug\" -> \"drug addiction\" [label=\"capable of\"];\n  \"drug addiction\" -> \"dangerous for society\" [label=\"is a\"];\n  \"dangerous for society\" -> \"legalized\" [label=\"not desires\"];\n}"
            }
        Return:
        class ExplaGraph:
            def __init__(self):
                belief = "Marijuana should not be legalized."
                argument = "Marijuana is dangerous for society."
                stance = "support"
                graph = "(marijuana; is a; recreational drug)(recreational drug; capable of; drug addiction)(drug addiction; is a; dangerous for society)(dangerous for society; not desires; legalized)"

        - If prompt part only is True, then the prompt part of the python code will be returned.
        """

        belief = row["belief"]
        argument = row["argument"]
        stance = row["stance"]

        py_source = ""
        py_source += f"class ExplaGraph:\n"
        py_source += f"    def __init__(self):\n"
        py_source += f"        self.belief = \"{belief}\"\n"
        py_source += f"        self.argument = \"{argument}\"\n"
        py_source += f"        self.stance = \"{stance}\"\n"
        
        if prompt_part_only:
            return py_source
        
        graph = row["graph"]
        py_source += f"        self.graph = \"{graph}\"\n"
        return py_source + "\n"

    def python_to_graph(self, py_code_str: str) -> str:
        """Given a python code string, generates a proscript schema string.

        Example:
        class ExplaGraph:
            def __init__(self):
                belief = "Marijuana should not be legalized."
                argument = "Marijuana is dangerous for society."
                stance = "support"
                graph = "(marijuana; is a; recreational drug)(recreational drug; capable of; drug addiction)(drug addiction; is a; dangerous for society)(dangerous for society; not desires; legalized)"


        returns:
            {
                "belief": "Marijuana should not be legalized.",
                "argument": "Marijuana is dangerous for society.",
                "stance": "support",
                "graph": "(marijuana; is a; recreational drug)(recreational drug; capable of; drug addiction)(drug addiction; is a; dangerous for society)(dangerous for society; not desires; legalized)",
                "question": "Belief: Marijuana should not be legalized., argument: Marijuana is dangerous for society.",
                "answer": "stance = support | digraph G {\n  \"marijuana\" -> \"recreational drug\" [label=\"is a\"];\n  \"recreational drug\" -> \"drug addiction\" [label=\"capable of\"];\n  \"drug addiction\" -> \"dangerous for society\" [label=\"is a\"];\n  \"dangerous for society\" -> \"legalized\" [label=\"not desires\"];\n}"
            }
        """
        print(py_code_str)
        # compile the code
        belief = re.search(r"belief = \"(.*?)\"", py_code_str).group(1)
        argument = re.search(r"argument = \"(.*?)\"", py_code_str).group(1)
        stance = re.search(r"stance = \"(.*?)\"", py_code_str).group(1)
        graph = re.search(r"graph = \"(.*?)\"", py_code_str).group(1)
        return {
            "belief": belief,
            "argument": argument,
            "stance": stance,
            "graph": graph,
        }

