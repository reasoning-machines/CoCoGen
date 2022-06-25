# retrieves examples from store
from typing import List
from sentence_transformers import util
import torch

from src.prompting.constants import END
from src.prompting.knnprompt.examplestore import ExampleStore

class Retriever(object):

    def get_closest(self, query: str, k: int, exclude_query: bool = True) -> List:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_closest, please use a subclass"
        )


class SemanticRetriever(Retriever):
    def __init__(
        self,
        example_store: ExampleStore
    ):
        
        self.example_store = example_store

    def get_closest(self, query: str, k: int) -> List:
        """
        Args:
            query (str): Input query
            k (int): Top k closest documents to the return
            most_relevant_last (bool, optional): If true, the documents are returned in ascending order of relevance.
                This is useful as the prompt should contain relevant information closest to the query.
        Returns:
            List[str]: [description]
        """


        scores = self.get_scores_for_query(self.example_store, query)

        top_k_idx = torch.topk(scores, k=min(k, len(scores)), dim=0)[1]

        matched_queries = []
        matched_documents = []
        for idx in top_k_idx:
            matched_documents.append(self.example_store.examples[idx][1])
            matched_queries.append(self.example_store.examples[idx][0])

        return matched_queries, matched_documents
    
    @staticmethod
    def get_scores_for_query(example_store: ExampleStore, query: str):

        # step 1: encode the query
        query_embedding = example_store.encode_query(query)

        # step 2: get the cosine similarity between the query and each document
        scores = util.cos_sim(query_embedding, example_store.query_embeddings).squeeze(0)

        return scores



class SemanticMultiRetriever(Retriever):
    def __init__(
        self,
        example_stores: List[ExampleStore]
    ):
        assert isinstance(example_stores, List)
        self.example_stores = example_stores

    def get_closest(self, query: str, k: int) -> List:
        """
        Args:
            query (str): Input query
            k (int): Top k closest documents to the return
            most_relevant_last (bool, optional): If true, the documents are returned in ascending order of relevance.
                This is useful as the prompt should contain relevant information closest to the query.
        Returns:
            List[str]: [description]
        """
        scores = []
        for i in range(len(self.example_stores)):
            scores.append(SemanticRetriever.get_scores_for_query(self.example_stores[i], query))
        
        scores = torch.stack(scores, dim=0).mean(dim=0)

        top_k_idx = torch.topk(scores, k=min(k, len(scores)), dim=0)[1]

        matched_queries = []
        matched_documents = []
        for idx in top_k_idx:
            #  just take the first store's example, as they are all the same
            matched_documents.append(self.example_stores[0].examples[idx][1])
            matched_queries.append(self.example_stores[0].examples[idx][0])

        return matched_queries, matched_documents
        
