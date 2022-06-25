from collections import Counter
import pickle
import sys
import numpy as np
from tqdm import tqdm
from typing import List, Tuple
import pydot
import networkx as nx


class PyDotMetrics:
    """PyDot metrics
    """
    @staticmethod
    def calc_num_nodes_num_edges(graphs: List[pydot.Graph], verbose: bool = False) -> Tuple[float, float]:
        num_nodes = []
        num_edges = []
        for g in tqdm(graphs, total=len(graphs), disable=not verbose):
            num_nodes.append(len(g.get_node_list()))
            num_edges.append(len(g.get_edge_list()))
        return np.mean(np.array(num_nodes)), np.mean(np.array(num_edges))


class NxMetrics:
    """Suite of graph related metrics. Most of them assume list of networkx graphs
    """
    @staticmethod
    def isomorphism(graph_a: List[nx.Graph], graph_b: List[nx.Graph], verbose: bool = True) -> float:
        isomorphism_count = 0
        for g_a, g_b in tqdm(zip(graph_a, graph_b), total=len(graph_a), disable=not verbose):
            if nx.is_isomorphic(g_a, g_b):
                isomorphism_count += 1
        return isomorphism_count / len(graph_a)

    @staticmethod
    def graph_edist(graph_a: List[nx.Graph], graph_b: List[nx.Graph], timeout: int, verbose: bool = True, return_arr: bool = False) -> Tuple[float, int]:
        total_edit = 0.
        num_edits_calculated = 0.
        edit_arr = []
        for g_a, g_b in tqdm(zip(graph_a, graph_b), total=len(graph_a), desc="Calculating edit distances", disable=not verbose):
            if g_a is None or g_b is None:
                continue
            edit_dist = nx.graph_edit_distance(g_a, g_b, timeout=timeout)
            total_edit += edit_dist
            num_edits_calculated += 1
            edit_arr.append(edit_dist)

        avg_edit = round(total_edit / num_edits_calculated, 3)
        if return_arr:
            return avg_edit, edit_arr
        return avg_edit, num_edits_calculated

    @staticmethod
    def calc_degree_summary_stats(graphs: List[nx.Graph], verbose: bool = False) -> Tuple[float, int]:
        avg_degrees = []
        all_degrees = []
        for g in tqdm(graphs, total=len(graphs), disable=not verbose):
            if g is None:
                continue
            degree_dict = dict(g.degree)
            if len(degree_dict) == 0:
                continue
            avg_degrees.append(sum(degree_dict.values()) / len(degree_dict))
            all_degrees.extend(degree_dict.values())

        avg_degrees = np.array(avg_degrees)
        all_degrees = np.array(all_degrees)
        return np.mean(avg_degrees),  np.median(avg_degrees), np.mean(all_degrees), np.median(all_degrees)

    @staticmethod
    def get_in_out_degree_distribution(graphs: List[nx.Graph], verbose: bool = False) -> Tuple[List[int], List[int]]:
        indegree_dist = []
        outdegree_dist = []
        degree_dist = []
        for g in tqdm(graphs, total=len(graphs), disable=not verbose):
            if g is None:
                continue
            indegree_dist.extend(dict(g.in_degree()).values())
            outdegree_dist.extend(dict(g.out_degree()).values())
            degree_dist.extend(dict(g.degree()).values())
        return Counter(indegree_dist), Counter(outdegree_dist), Counter(degree_dist)
