"""
Each example is of the following form:

{
    "n": 4,
    "binary_num": "1010110000110011",
    "func_cnf": "And(Or(~x[0], ~x[1], x[3]), Or(~x[1], ~x[2], x[3]), Or(~x[0], x[1], x[2]), Or(~x[0], x[2], x[3]), Or(x[1], ~x[3]))",
    "list_of_equations_cnf": [
        "var0 = ~x[0] | ~x[1] | x[3]",
        "var1 = ~x[1] | ~x[2] | x[3]",
        "var2 = ~x[0] | x[1] | x[2]",
        "var3 = ~x[0] | x[2] | x[3]",
        "var4 = x[1] | ~x[3]",
        "var5 = var0 & var1 & var2 & var3 & var4"
    ],
    "verilog_code_cnf": "module test(x, y);\ninput [3:0] x;\noutput y;\nwire var0, var1, var2, var3, var4, var5;\nassign var0 = ~x[0] | ~x[1] | x[3];\nassign var1 = ~x[1] | ~x[2] | x[3];\nassign var2 = ~x[0] | x[1] | x[2];\nassign var3 = ~x[0] | x[2] | x[3];\nassign var4 = x[1] | ~x[3];\nassign var5 = var0 & var1 & var2 & var3 & var4;\nassign y = var5;\nendmodule\n",
    "test_bench_code_cnf": "`include \"src/1010110000110011.v\"\n`include \"test/assert.v\"\nmodule test_bench;\nreg [3:0] x;\nwire y;\nreg [0:15] true_y = 16'b1010110000110011;\ninteger i;\ntest uut(x, y);\ninitial begin\n\tfor (i = 0; i < 16; i = i + 1) begin\n\t\tx = i;\n\t\t# 1;\n\t\t`assert(y, true_y[i]);\n\t\t$display(\"x=%b, y=%b, true_y=%b\", x, y, true_y[i]);\n\tend\nend\nendmodule\n",
    "canonical_type_cnf": "cnf",
    "func_dnf": "Or(And(x[1], x[3]), And(~x[0], ~x[2], ~x[3]), And(~x[1], x[2], ~x[3]))",
    "list_of_equations_dnf": [
        "var0 = x[1] & x[3]",
        "var1 = ~x[0] & ~x[2] & ~x[3]",
        "var2 = ~x[1] & x[2] & ~x[3]",
        "var3 = var0 | var1 | var2"
    ],
    "verilog_code_dnf": "module test(x, y);\ninput [3:0] x;\noutput y;\nwire var0, var1, var2, var3;\nassign var0 = x[1] & x[3];\nassign var1 = ~x[0] & ~x[2] & ~x[3];\nassign var2 = ~x[1] & x[2] & ~x[3];\nassign var3 = var0 | var1 | var2;\nassign y = var3;\nendmodule\n",
    "test_bench_code_dnf": "`include \"src/1010110000110011.v\"\n`include \"test/assert.v\"\nmodule test_bench;\nreg [3:0] x;\nwire y;\nreg [0:15] true_y = 16'b1010110000110011;\ninteger i;\ntest uut(x, y);\ninitial begin\n\tfor (i = 0; i < 16; i = i + 1) begin\n\t\tx = i;\n\t\t# 1;\n\t\t`assert(y, true_y[i]);\n\t\t$display(\"x=%b, y=%b, true_y=%b\", x, y, true_y[i]);\n\tend\nend\nendmodule\n",
    "canonical_type_dnf": "dnf",
    "verilog_code_cnf_len": 330,
    "verilog_code_dnf_len": 234,
    "cnf_dnf_eq": 0,
    "input": "module test(x, y);\ninput [3:0] x;\noutput y;\nwire var0, var1, var2, var3, var4, var5;\nassign var0 = ~x[0] | ~x[1] | x[3];\nassign var1 = ~x[1] | ~x[2] | x[3];\nassign var2 = ~x[0] | x[1] | x[2];\nassign var3 = ~x[0] | x[2] | x[3];\nassign var4 = x[1] | ~x[3];\nassign var5 = var0 & var1 & var2 & var3 & var4;\nassign y = var5;\nendmodule\n",
    "output": "module test(x, y);\ninput [3:0] x;\noutput y;\nwire var0, var1, var2, var3;\nassign var0 = x[1] & x[3];\nassign var1 = ~x[0] & ~x[2] & ~x[3];\nassign var2 = ~x[1] & x[2] & ~x[3];\nassign var3 = var0 | var1 | var2;\nassign y = var3;\nendmodule\n",
    "meta": {
        "func": "Or(And(x[1], x[3]), And(~x[0], ~x[2], ~x[3]), And(~x[1], x[2], ~x[3]))",
        "list_of_equations": [
            "var0 = x[1] & x[3]",
            "var1 = ~x[0] & ~x[2] & ~x[3]",
            "var2 = ~x[1] & x[2] & ~x[3]",
            "var3 = var0 | var1 | var2"
        ],
        "verilog_code": "module test(x, y);\ninput [3:0] x;\noutput y;\nwire var0, var1, var2, var3;\nassign var0 = x[1] & x[3];\nassign var1 = ~x[0] & ~x[2] & ~x[3];\nassign var2 = ~x[1] & x[2] & ~x[3];\nassign var3 = var0 | var1 | var2;\nassign y = var3;\nendmodule\n",
        "test_bench_code": "`include \"src/1010110000110011.v\"\n`include \"test/assert.v\"\nmodule test_bench;\nreg [3:0] x;\nwire y;\nreg [0:15] true_y = 16'b1010110000110011;\ninteger i;\ntest uut(x, y);\ninitial begin\n\tfor (i = 0; i < 16; i = i + 1) begin\n\t\tx = i;\n\t\t# 1;\n\t\t`assert(y, true_y[i]);\n\t\t$display(\"x=%b, y=%b, true_y=%b\", x, y, true_y[i]);\n\tend\nend\nendmodule\n",
        "canonical_type": "dnf"
    }
}

Our goal is converting CNF to DNF.
"""

import re
from src.converters.graph_code_converter import GraphPythonConverter

LONGER = "CNF"
SHORTER = "Shorter DNF"

class BoolgenPythonConverterLiteral(GraphPythonConverter):

    def graph_to_python(self, row: dict, prompt_part_only: bool) -> str:
        """Converts a boolgen example to one used in the prompt.
        Example:

        Return: 
        "Longer code:
            row['verilog_code_cnf']
        Shorter code:
            row['verilog_code_dnf']
        - If prompt part only is True, then only row['verilog_code_cnf'] is returned.
        """

        cnf_code = row['verilog_code_cnf']
        dnf_code = row['verilog_code_dnf']
        # The below is only done for train
        cnf_code_len = len(cnf_code.split())
        dnf_code_len = len(dnf_code.split())
        if cnf_code_len < dnf_code_len:
            return None
        perc_diff = (cnf_code_len - dnf_code_len) / cnf_code_len * 100
        if perc_diff < 25:
            return None

        py_source = f"{LONGER} code:\n\n" + cnf_code
        if prompt_part_only:
            return py_source
        
        py_source += f"\n\n{SHORTER} code:\n\n" + dnf_code + "\n"
        return py_source

    def python_to_graph(self, py_code_str: str) -> str:
        """Longer code:
           row['verilog_code_cnf']
           
           Shorter code:
           row['verilog_code_dnf']
        """
        cnf_code = re.search(fr'{LONGER} code:\n\n(.*?)\n\n{SHORTER} code:', py_code_str, re.DOTALL).group(1)
        dnf_code = re.search(fr'{SHORTER} code:\n\n(.*?)\n\n', py_code_str, re.DOTALL).group(1)
        return {
            'verilog_code_cnf': cnf_code,
            'verilog_code_dnf': dnf_code,
        }

