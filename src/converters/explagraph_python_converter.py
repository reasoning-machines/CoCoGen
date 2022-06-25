
import pandas as pd

from src.converters.utils import to_snake_case
from src.converters.graph_code_converter import GraphPythonConverter

class ExplagraphPythonConverter(GraphPythonConverter):
    def __init__(self):
        super().__init__()

    def graph_to_python(self, row: dict, prompt_part_only: bool) -> str:
        """Given a row that contains 
        - belief = "some belief", 
        - argument = "some argument",
        - stance = "some stance",
        - graph = "(node1; rel1; node2)(node3; rel2; node4)(node5; rel3; node6)"

        1. First, extracts the relations from the graph in a dictionary.
            The relations are in the form:
            * {
                "rel1": [(node1, node2)],
                "rel2": [(node3, node4), (node5, node6)]
            }
            * where node1 and node2 are the nodes that are connected by the relation

        2. Next, creates a python class, where belief, argument, and stance are attributes.
        - Each relation is a function that takes in two nodes and returns a boolean.
        - The relations are listed in a main function which list assertions for each relation.

            class BeliefArgument(object):
                belief = "some belief"
                argument = "some argument"
                stance = "some stance"

                self.nodes = {node1, node2, node3, node4, node5, node6}

                def rel1(self, node1, node2):
                    return check_relation("rel1", node1, node2)
                
                def rel2(self, node1, node2):
                    return check_relation("rel2", node1, node2)
                
                def main(self):
                    relations = {
                        f"{node}
                    }
        """
        print(row)
        belief, argument = row["belief"], row["argument"]
        stance = row["stance"]
        relations = self.get_relations(row)
        nodes = self.get_nodes(relations)

        python_src = ""
        python_src += f"class BeliefArgument:\n\n"

        # add init
        python_src += f"    def __init__(self):\n"
        # add belief, argument, stance
        python_src += f"        self.belief = \"{belief}\"\n"
        python_src += f"        self.argument = \"{argument}\"\n\n"
        python_src += f"        # add stance\n"
        python_src += f"        self.stance = \"{stance}\"\n\n"
        

        # add nodes
        
        python_src += f"        # add nodes\n"
        if prompt_part_only:
            return python_src

        python_src += f"        self.nodes = {nodes}\n\n"

        # add relations
        
        for rel_name, rel_nodes in relations.items():
            rel_name = to_snake_case(rel_name)
            python_src += f"    def {rel_name}(self, node1, node2):\n"
            python_src += f"        return check_relation(\"{rel_name}\", node1, node2)\n\n"
        
        # add main
        graph_str = row["graph"]
        python_src += f"    def graph_str(self):\n"
        python_src += f"        return \"{graph_str}\"\n\n"  
        python_src += f"\n"

        return python_src

    def python_to_graph(self, py_code_str: str) -> str:
        """Given a python code string, extracts the corresponding explagraph string

        class BeliefArgument(object):

            def __init__(self):
                self.belief = "People should be able to do missionary work if they desire."
                self.argument = "People should have right to missionary work."

                # add stance
                self.stance = "support"

                # now add relation asserts
                assert self.desires("people", "volunteer_opportunities")
                assert self.capable_of("volunteer_opportunities", "missionary_work")
                assert self.is_a("missionary_work", "right")
        """
        # compile the code
        py_code = compile(py_code_str, "<string>", "exec")

        # instantiate the class
        py_code_dict = {}
        exec(py_code, py_code_dict)
        py_code_class = py_code_dict[list(py_code_dict.keys())[1]]()
        return py_code_class.graph_str()


    def get_relations(self, row):
        graph = row["graph"]
        edges = graph.split("(")[1:]
        relations = dict()
        for edge in edges:
            edge = edge.strip()
            node1, rel, node2 = edge.split(";")
            node1 = node1.strip().replace("(", "").replace(")", "")
            node2 = node2.strip().replace("(", "").replace(")", "")
            rel = rel.strip().replace("(", "").replace(")", "")
            if rel not in relations:
                relations[rel] = [(node1, node2)]
            else:
                relations[rel].append((node1, node2))
        
        return relations

    def get_nodes(self, relations):
        all_nodes = set()
        for rel, nodes in relations.items():
            for (node1, node2) in nodes:
                all_nodes.add(node1)
                all_nodes.add(node2)
        return all_nodes



def run(inpath: str, outpath: str):
    converter = ExplagraphPythonConverter()
    data = pd.read_csv(inpath, sep="\t", header=None, names=["belief", "argument", "stance", "graph"])
    
    data["py_source"] = data.apply(converter.graph_to_python, axis=1)
    data.to_json(outpath, orient='records', lines=True)

    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--inpath", type=str, required=True)
    parser.add_argument("--outpath", type=str, required=True)
    args = parser.parse_args()
    run(args.inpath, args.outpath)
    print("Done.")