from converters.proscript.edge_pred.edge_pred_task import ProscriptPythonConverterMethodEdgePredTask
from converters.proscript.literal import ProscriptPythonConverterLiteral
from converters.proscript.networkx import ProscriptPythonConverterNetworkx
from converters.proscript.hashmap_sep import ProscriptPythonConverterHashmapSep
from converters.proscript.hashmap_init import ProscriptPythonConverterHashmapInit
from converters.proscript.oo import ProscriptPythonConverterOO
from converters.proscript.text import ProscriptPythonConverterText
from converters.proscript.method import ProscriptPythonConverterMethod
from converters.proscript.method_explicit import ProscriptPythonConverterMethodExplicit
from converters.proscript.edge_pred.edge_pred_literal import ProscriptPythonConverterEdgePredLiteral
from converters.proscript.edge_pred.edge_pred_networkx import ProscriptPythonConverterEdgePredNetworkx
from converters.proscript.edge_pred.edge_pred_dep import ProscriptPythonConverterMethodEdgePredDep
from converters.proscript.edge_pred.edge_pred_dep_succ import ProscriptPythonConverterMethodEdgePredDepSucc
from converters.proscript.edge_pred.edge_pred_dep_succ_lined import ProscriptPythonConverterMethodEdgePredDepSuccLined
from converters.proscript.edge_pred.edge_pred_assert import ProscriptPythonConverterMethodEdgePredAssert
from converters.proscript.edge_pred.edge_pred_tree import ProscriptPythonConverterMethodEdgePredTree
from converters.proscript.edge_pred.edge_pred_tree_begin import ProscriptPythonConverterMethodEdgePredTreeBegin
from converters.proscript.edge_pred.edge_pred_node_list import ProscriptPythonConverterMethodEdgePredNodeList
from converters.proscript.edge_pred.edge_pred_dep_rand_inst import ProscriptPythonConverterMethodEdgePredDepRandInst
from converters.proscript.edge_pred.edge_pred_direct import ProscriptPythonConverterEdgePredDirect
from converters.proscript.edge_pred.edge_pred_direct_inst import ProscriptPythonConverterEdgePredDirectInst
from converters.proscript.edge_pred.edge_pred_text import ProscriptPythonConverterEdgePredText
from converters.proscript.edge_pred.edge_pred_event import ProscriptPythonConverterMethodEdgePredEvent
from converters.proscript.edge_pred.edge_pred_dep_fixer import ProscriptPythonConverterMethodEdgePredDepFixer
from converters.proscript.edge_pred.edge_pred_tree_inst import ProscriptPythonConverterMethodEdgePredTreeInst
from converters.proscript.edge_pred.edge_pred_nodet import ProscriptPythonConverterMethodEdgePredNoDet
from converters.proscript.edge_pred.edge_pred_nodet_fixer import ProscriptPythonConverterMethodEdgeNoDetFixer
from converters.proscript.tree import ProscriptPythonConverterMethodTree


from converters.explagraphs.tree import ExplagraphPythonConverterTree
from converters.explagraphs.tree_edge_only import ExplagraphPythonConverterTreeEdgeOnly
from converters.explagraphs.relation import ExplagraphPythonConverterRelation
from converters.explagraph_python_converter import ExplagraphPythonConverter
from converters.explagraphs.literal import ExplagraphPythonConverterLiteral
from converters.explagraphs.text import ExplagraphPythonConverterText

<<<<<<< HEAD
from converters.boolgen.literal import BoolgenPythonConverterLiteral

# from converters.propara.func import ProparaPythonConverterFunc, ProparaPythonConverterClass
=======
from converters.propara.func import ProparaPythonConverterFunc, ProparaPythonConverterClass, ProparaNLConverterQA
>>>>>>> a8e3d47129dd1da557095732e4a01f261208a517

class ConverterFactory:

    converter_to_class = {
        "proscript-hashmap-init": ProscriptPythonConverterHashmapInit,
        "proscript-hashmap-sep": ProscriptPythonConverterHashmapSep,
        "proscript-networkx": ProscriptPythonConverterNetworkx,
        "proscript-literal": ProscriptPythonConverterLiteral,
        "proscript-oo": ProscriptPythonConverterOO,
        "proscript-text": ProscriptPythonConverterText,
        "proscript-method": ProscriptPythonConverterMethod,
        "proscript-tree": ProscriptPythonConverterMethodTree,
        "proscript-method-explicit": ProscriptPythonConverterMethodExplicit,

        "proscript-edge-pred-literal": ProscriptPythonConverterEdgePredLiteral,
        "proscript-edge-pred-networkx": ProscriptPythonConverterEdgePredNetworkx,
        "proscript-edge-pred-dep": ProscriptPythonConverterMethodEdgePredDep,
        "proscript-edge-pred-dep-succ": ProscriptPythonConverterMethodEdgePredDepSucc,
        "proscript-edge-pred-dep-succ-lined": ProscriptPythonConverterMethodEdgePredDepSuccLined,
        "proscript-edge-pred-direct": ProscriptPythonConverterEdgePredDirect,
        "proscript-edge-pred-direct-inst": ProscriptPythonConverterEdgePredDirectInst,
        "proscript-edge-pred-dep-rand-inst": ProscriptPythonConverterMethodEdgePredDepRandInst,
        "proscript-edge-pred-tree": ProscriptPythonConverterMethodEdgePredTree,
        "proscript-edge-pred-tree-begin": ProscriptPythonConverterMethodEdgePredTreeBegin,
        "proscript-edge-pred-node-list": ProscriptPythonConverterMethodEdgePredNodeList,
        "proscript-edge-pred-tree-inst": ProscriptPythonConverterMethodEdgePredTreeInst,
        "proscript-edge-pred-event": ProscriptPythonConverterMethodEdgePredEvent,
        "proscript-edge-pred-text": ProscriptPythonConverterEdgePredText,
        "proscript-edge-pred-dep-fixer": ProscriptPythonConverterMethodEdgePredDepFixer,
        "proscript-edge-pred-nodet-fixer": ProscriptPythonConverterMethodEdgeNoDetFixer,
        "proscript-edge-pred-assert": ProscriptPythonConverterMethodEdgePredAssert,
        "proscript-edge-pred-nodet": ProscriptPythonConverterMethodEdgePredNoDet,
        "proscript-edge-pred-task": ProscriptPythonConverterMethodEdgePredTask,

        "explagraphs": ExplagraphPythonConverter,
        "explagraphs-literal": ExplagraphPythonConverterLiteral,
        "explagraphs-tree": ExplagraphPythonConverterTree,
        "explagraphs-tree-edge-only": ExplagraphPythonConverterTreeEdgeOnly,
        "explagraphs-relation": ExplagraphPythonConverterRelation,
        "explagraphs-text": ExplagraphPythonConverterText,

        "boolgen-literal": BoolgenPythonConverterLiteral,

        "propara-func": ProparaPythonConverterFunc,
        "propara-class": ProparaPythonConverterClass,
        "propara-text-qa": ProparaNLConverterQA
    }
    supported_converters = list(converter_to_class.keys())

    @staticmethod
    def get_converter(job_type: str):
        if job_type not in ConverterFactory.supported_converters:
            raise ValueError(f"Unsupported job type: {job_type}")
        return ConverterFactory.converter_to_class[job_type]()
