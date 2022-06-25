class GraphPythonConverter(object):

    def graph_to_python(self, graph_str: str, prompt_part_only: bool) -> str:
        """Placeholder for converting graph string to code.
        Should be implemented by subclass.
        Args:
            graph_str: A graph string.

        Returns:
            A string containing the code.
        """
        raise NotImplementedError()
    
    def python_to_graph(self, code_str: str) -> str:
        """Placeholder for converting code to graph string.
        Should be implemented by subclass.
        Args:
            code_str: A code string.

        Returns:
            A string containing the graph.
        """
        raise NotImplementedError()
