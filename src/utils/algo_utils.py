import networkx as nx
from typing import List, Dict

class GraphAlgos:

    @staticmethod
    def topo_sort(graph: nx.DiGraph) -> List:
        return list(nx.topological_sort(graph))
    
    @staticmethod
    def root_nodes(graph: nx.DiGraph) -> List:
        """Given a graph in adjacency list format, returns
        a list of root nodes.
        """
        # find all nodes with in-degree of 0
        return [node for node in graph if graph.in_degree(node) == 0]

    @staticmethod
    def to_nx_graph(src_dest_dict: Dict) -> nx.DiGraph:
        """Given a graph in adjacency list format, returns
        a networkx graph.
        """
        graph = nx.DiGraph()
        for src, dests in src_dest_dict.items():
            for dest in dests:
                graph.add_edge(src, dest)
        return graph
    
    @staticmethod
    def get_predecessors(graph: nx.DiGraph, node: str) -> List:
        return list(graph.predecessors(node))

    @staticmethod
    def get_successors(graph: nx.DiGraph, node: str) -> List:
        return list(graph.successors(node))