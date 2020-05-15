import json
from .helpers import dump
from datetime import datetime

class RetrievalScore:
    def __init__(self):
        self.retrieval_score = None
        self.is_evaluated = False

    def estimate_sparsity(self, repres):
        zero = 0
        for i in range(len(repres)):
            if repres[i] == 0.0:
                zero += 1
        return zero
        
    """
        This function returns a dictionary, where each query has a correspondent
        relative documents with the corresponding score.
    """

    def evaluate(self, eval_loader, index, model, batch_size):
        start = datetime.now()
        if self.is_evaluated:
            return self.retrieval_score
        is_end = False
        self.retrieval_score = dict()
        counter = 0
        while not is_end:
            queries_id, queries, is_end = eval_loader.generate_queries(batch_size)
            qreprs = model.evaluate_repr(queries, input_type="queries")
            for qrepr, q in zip(qreprs, queries_id):
                if counter < 3:
                    print("Zero elements: ", self.estimate_sparsity(qrepr), len(qrepr), flush=True)
                counter += 1
                self.retrieval_score[str(q)] = self.__retrieval_score_for_query(
                    qrepr, index
                )  # returns dict({doc_id:val})
        time = datetime.now() - start
        print("Retrieval score is built for ", time, flush = True)
        self.is_evaluated = True
        return self.retrieval_score

    """
        This function estimates retrieval score for each query
        over all possible documents from the inverted index.
        
        Returns a dictionaty of shape:
        {
            'doc1_id': val,
            'doc2_id': val
        }
    """

    def __retrieval_score_for_query(self, query_repr, index):
        relevant_docs = dict()
        for i in range(len(query_repr)):
            if query_repr[i] != 0.0:
                try:
                    docs = index.get_index()[i]
                except KeyError:
                    continue
                docs = index.get_index()[i]
                for j in range(len(docs)):
                    doc_id = str(docs[j][0])
                    if doc_id not in relevant_docs:
                        relevant_docs[doc_id] = query_repr[i].item() * docs[j][1]
                    else:
                        relevant_docs[doc_id] += query_repr[i].item() * docs[j][1]
        return relevant_docs

    """
        Dump retrieval score.
    """

    def dump(self, filename):
        if not self.is_evaluated:
            print("Retrieval score is not yet evaluated")
        else:
            dump(self.retrieval_score, filename)

    """
        Read retrieval score to dict.
    """

    def read(self, filename):
        if self.is_evaluated:
            print("One retrieval score is already loaded")
        else:
            with open(filename, "r") as f:
                self.retrieval_score = json.load(f)
        return self.retrieval_score
