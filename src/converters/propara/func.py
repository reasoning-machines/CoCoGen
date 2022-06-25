import re

from src.converters.graph_code_converter import GraphPythonConverter
from src.converters.propara.function_variable import CodePromptCreator
from src.converters.propara.class_attribute import CodePromptCreatorV2
from src.converters.propara.natural_language_qa import NLPromptCreator

class ProparaPythonConverterFunc(GraphPythonConverter):

    def __init__(self) -> None:
        super().__init__()
        self.converter = CodePromptCreator(oracle_initial_state=False)

    def graph_to_python(self, row: dict, prompt_part_only: bool) -> str:
        """Converts a proscript schema to a python class.
        """
        if prompt_part_only:
            return self.converter.generate_sample_head(row) + "\n"
        return self.converter.paragraph_to_code(row) + "\n"

    def python_to_graph(self, py_code_str: str) -> str:
        """For now, this is a no-op.
        """
        return py_code_str + "\n"


class ProparaPythonConverterClass(GraphPythonConverter):

    def __init__(self) -> None:
        super().__init__()
        self.converter = CodePromptCreatorV2()

    def graph_to_python(self, row: dict, prompt_part_only: bool) -> str:
        """Converts a proscript schema to a python class.
        """
        if prompt_part_only:
            return self.converter.generate_sample_head(row) + "\n"
        return self.converter.paragraph_to_code(row) + "\n"

    def python_to_graph(self, py_code_str: str) -> str:
        """For now, this is a no-op.
        """
        return py_code_str + "\n"

class ProparaNLConverterQA(GraphPythonConverter):

    def __init__(self) -> None:
        super().__init__()
        self.converter = NLPromptCreator()

    def graph_to_python(self, row: dict, prompt_part_only: bool) -> str:
        """Converts a proscript schema to a python class.
        """
        if prompt_part_only:
            return self.converter.generate_sample_head(row) + "\n"
        return self.converter.paragraph_to_code(row) + "\n"

    def python_to_graph(self, py_code_str: str) -> str:
        """For now, this is a no-op.
        """
        return py_code_str + "\n"