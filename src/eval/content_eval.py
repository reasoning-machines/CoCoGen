from typing import Dict, List, Union
from typing import Dict, List, Union
import re
import pandas as pd
import pprint

from gem_metrics.texts import Predictions, References
from gem_metrics.local_recall import LocalRecall
from gem_metrics.chrf import CHRF
from gem_metrics.bleu import BLEU
from gem_metrics.nist import NIST
from gem_metrics.rouge import ROUGE
from gem_metrics.meteor import Meteor

from gem_metrics.bertscore import BERTScore
from gem_metrics.bleurt import BLEURT

# referenceless_metrics
from gem_metrics.ngrams import NGramStats
from gem_metrics.msttr import MSTTR
import logging

import tensorflow as tf
import time

logging.basicConfig(level=logging.INFO)
gpus = tf.config.list_physical_devices("GPU")

if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices("GPU")
        logging.info(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        logging.info(e)




class NodeMetricsCalculator:
    def __init__(self) -> None:
        self.rouge = ROUGE()
        self.bleu = BLEU()
        self.chrf = CHRF()
        self.nist = NIST()
        self.local_recall = LocalRecall()
        self.ngram_stats = NGramStats()
        self.msttr = MSTTR()
        self.meteor = Meteor()
        # load bertscore on the first gpu
        with tf.device("/job:localhost/replica:0/task:0/device:GPU:1"):
            self.bert_scorer = BERTScore()

        # load bleurt on the second gpu
        with tf.device("/job:localhost/replica:0/task:0/device:GPU:3"):
            self.bleurt_scorer = BLEURT()

    def score_from_text_list(self, predictions: List[str], references: List[str]) -> dict:
        predictions = self.get_predictions(predictions)
        references = self.get_references(references)
        assert len(references) == len(predictions)
        predictions.assign_ids_and_unscramble(id_list=references.ids)
        return self.score(predictions, references)
        
    def score(self, predictions: Predictions, references: References) -> dict:


        ngram_stats_score = self.run_metric(
            metric_name="NGram Stats",
            predictions=predictions,
            scorer=self.ngram_stats,
            references=None
        )


        pred_msttr_score = self.run_metric(metric_name="MSTTR", predictions=predictions, references=None, scorer=self.msttr)
        ref_msttr_score = self.run_metric(metric_name="MSTTR", predictions=references, references=None, scorer=self.msttr)


        bleurt_score = self.run_metric(
            metric_name="BLEURT",
            predictions=predictions,
            references=references,
            scorer=self.bleurt_scorer,
        )["bleurt"]

        bert_score = self.run_metric(
            metric_name="BERTScore",
            predictions=predictions,
            references=references,
            scorer=self.bert_scorer,
        )["bertscore"]["f1"]
        # bleurt_score = 0
        # bert_score = 0

        bleu_score = self.run_metric(
            metric_name="BLEU", predictions=predictions, references=references, scorer=self.bleu
        )["bleu"]

        rouge_score = self.run_metric(
            metric_name="ROUGE", predictions=predictions, references=references, scorer=self.rouge
        )
        rougeL = rouge_score["rougeL"]["fmeasure"]
        rouge2 = rouge_score["rouge2"]["fmeasure"]

        nist_score = self.run_metric(
            metric_name="NIST", predictions=predictions, references=references, scorer=self.nist
        )["nist"]

        chrf_score = self.run_metric(
            metric_name="CHRF", predictions=predictions, references=references, scorer=self.chrf
        )["chrf"]

        local_recall_score = self.run_metric(
            metric_name="Local Recall",
            predictions=predictions,
            references=references,
            scorer=self.local_recall,
        )["local_recall"]

        return {
            "num_samples": len(predictions),
            "bleu": bleu_score,
            "rougeL": rougeL,
            "rouge2": rouge2,
            # "meteor": meteor_score,
            "bertscore": bert_score,
            "bleurt": bleurt_score,
            "nist": nist_score,
            "chrf": chrf_score,
            "local_recall": local_recall_score,
            "ngram_stats": ngram_stats_score,
            "pred_msttr": pred_msttr_score,
            "ref_msttr": ref_msttr_score,
        }



    def run_metric(self, metric_name, scorer, predictions, references):
        logging.info(f"Calculating {metric_name}")
        time_before = time.time()
        if references is None:
            score = scorer.compute_cached(None, predictions=predictions)
        else:
            score = scorer.compute_cached(None, predictions, references)
        time_after = time.time()
        elapsed_time = round(time_after - time_before)
        logging.info(f"{metric_name} took {elapsed_time} seconds")
        return score



    def get_predictions(self, predictions):
        data = {"values": predictions, "language": "en"}
        return Predictions(data=data)


    def get_references(self, references):
        references = [{"target": [ref]} for ref in references]
        data = {"values": references, "language": "en"}
        return References(data=data)


    def run(self, data: pd.DataFrame, ref_col, pred_col, get_nodes_func):
        """Given path to a file with outputs, reference column, and prediction column,
        and a function that can extract nodes from both the graphs, computes generation metrics.

        Args:
            path (str): _description_
            ref_col (_type_): _description_
            pred_col (_type_): _description_
            get_nodes_func (_type_): _description_
        """
        
        predicted_nodes = data[pred_col].apply(lambda x: get_nodes_func(x)).tolist()
        reference_nodes = data[ref_col].apply(lambda x: get_nodes_func(x)).tolist()

        logging.info(f"Found {len(predicted_nodes)} predictions")
        # print 5 examples
        for i in range(5):
            logging.info(f"Predicted: {predicted_nodes[i]} vs. Reference: {reference_nodes[i]}")

        report = self.score_from_text_list(predicted_nodes, reference_nodes)
        return report


