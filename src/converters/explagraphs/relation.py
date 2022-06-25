import re
from typing import List
from src.converters.graph_code_converter import GraphPythonConverter
from utils.algo_utils import GraphAlgos
from src.converters.utils import to_camel_case, to_snake_case, from_snake_to_normal_str


class ExplagraphPythonConverterRelation(GraphPythonConverter):

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

                # tree for argument in support of belief

                marijuana = Node()
                marijuana.add_edge("is a", "recreational drug")
                marijuana.add_edge("capable of", "drug addiction")
                recreational_drug = Node()
                recreational_drug.add_edge("is a", "drug addiction")
                recreational_drug.add_edge("capable of", "drug addiction")
                drug_addiction = Node()
                drug_addiction.add_edge("is a", "dangerous for society")
                dangerous_for_society = Node()
                dangerous_for_society.add_edge("not desires", "legalized")




        - If prompt part only is True, then the prompt part of the python code will be returned.
        """

        belief = row["belief"]
        argument = row["argument"]
        stance = row["stance"]

        py_source = ""
        py_source += f"class ExplanationDAG:\n\n"
        py_source += f"    def __init__(self):\n"
        py_source += f"        belief = \"{belief}\"\n"
        py_source += f"        argument = \"{argument}\"\n"
        py_source += f"        stance = \"{stance}\"\n"

        py_source += "\n"
        py_source += f"        # create a DAG to {stance} belief using argument\n"

        if prompt_part_only:
            return py_source
        
        
        graph = row["graph"]
        nodes, from_to_dict, from_to_dict_typed = self.get_nodes(graph)

        steps_graph = GraphAlgos.to_nx_graph(from_to_dict)
        root_nodes = GraphAlgos.root_nodes(steps_graph)
        root_nodes_str = ", ".join([f"\"{node}\"" for node in root_nodes])
        py_source += f"        begin = [{root_nodes_str}]\n"
        for node in from_to_dict_typed:
            edges = from_to_dict_typed[node]
                # py_source += f"        {node} = Node()\n"
            for (edge_type, edge_to) in edges:
                py_source += f"        add_edge(\"{node}\", \"{edge_type}\", \"{edge_to}\")\n"

        return py_source + "\n"

    def get_nodes(self, graph_str) -> List[str]:
        """Returns a list of nodes in the graph.

        Example:
        "(marijuana; is a; recreational drug)(recreational drug; capable of; drug addiction)(drug addiction; is a; dangerous for society)(dangerous for society; not desires; legalized)"

        returns:
        ["marijuana", "recreational drug", "drug addiction", "dangerous for society", "legalized"]

        """
        nodes = []
        from_to_dict = dict()
        from_to_dict_typed = dict()
        graph_str = graph_str[1:-1]
        graph_parts = graph_str.split(")(")
        for edge in graph_parts:
            if ";" in edge:
                from_node, edge_type, to_node = edge.split(";")
                from_node = from_node.strip()
                to_node = to_node.strip()
                edge_type = edge_type.strip()
                if from_node not in nodes:
                    nodes.append(from_node)
                if to_node not in nodes:
                    nodes.append(to_node)
                if from_node not in from_to_dict:
                    from_to_dict[from_node] = []
                    from_to_dict_typed[from_node] = []
                if to_node not in from_to_dict[from_node]:
                    from_to_dict[from_node].append(to_node)
                    from_to_dict_typed[from_node].append((edge_type, to_node))

        return nodes, from_to_dict, from_to_dict_typed




    def python_to_graph(self, py_code_str: str) -> str:
        """Given a python code string, generates a proscript schema string.

        Example:
        class ExplaGraph:
            def __init__(self):
                belief = "Marijuana should not be legalized."
                argument = "Marijuana is dangerous for society."
                stance = "support"
                marijuana = Node()
                marijuana.add_edge("is a", "recreational drug")
                marijuana.add_edge("capable of", "drug addiction")
                recreational_drug = Node()
                recreational_drug.add_edge("is a", "drug addiction")
                recreational_drug.add_edge("capable of", "drug addiction")
                drug_addiction = Node()
                drug_addiction.add_edge("is a", "dangerous for society")
                dangerous_for_society = Node()
                dangerous_for_society.add_edge("not desires", "legalized")


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
        # compile the code
        belief = re.search(r"belief = \"(.*?)\"", py_code_str).group(1)
        argument = re.search(r"argument = \"(.*?)\"", py_code_str).group(1)
        stance = re.search(r"stance = \"(.*?)\"", py_code_str).group(1)
        code_lines = py_code_str.split("\n")
        graph_str = ""
        for line in code_lines:
            if "add_edge" in line:
                line = line.split("(")[1][:-1]
                fields = line.split(", \"")
                source = fields[0].strip()
                edge_type = fields[1].strip()
                target = fields[2].strip()

                # edge_type = re.search(r"\"(.*?)\"", line).group(1)
                # target = re.search(r"\"(.*?)\"", line).group(2)
                graph_str += f"({source}; {edge_type}; {target})"
        
        graph_str = graph_str.replace("\"", "")
        graph_str = graph_str.replace(";", "; ")
        # graph_str = graph_str.replace("(", " (")
        # graph_str = graph_str.replace(")", ") ")
        graph_str = graph_str.replace("  ", " ")
        print(py_code_str)
        return {
            "belief": belief,
            "argument": argument,
            "stance": stance,
            "graph": graph_str,
        }
                


