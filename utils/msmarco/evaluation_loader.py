import pandas as pd
import numpy as np
from ..pytrec_evaluator import read_qrels

"""
    EvaluationLoader is used to generate docs and queries batches
    and to get the relevant docs by query.
"""


class EvaluationLoader:
    def __init__(
        self, docs, queries=None, qrels=None, rs=0, df_queries=None
    ):
        self.doc_offset = 0
        self.query_offset = 0
        self.qrels = qrels
        
        self.__init_df(docs, queries, df_queries)

    def __init_df(self, docs, queries, df_queries):
       self.docs_file = open(docs, "rt", encoding="utf8")
       self.docs_len = sum(1 for line in self.docs_file)
       
       if df_queries is None:
           self.df_queries = pd.read_csv(queries, header=None, sep="\t", na_filter=False)
       else:
           self.df_queries = df_queries
       self.queries_len = self.df_queries.shape[0]

    def docs_length(self):
        return self.docs_len

    def queries_length(self):
        return self.queries_len

    """
        Returns the query batch.
    """

    def generate_queries(self, size=256):
        query_ids = []
        queries = []
        end = min(self.query_offset + size, self.queries_len)
        for i in range(self.query_offset, end):
            query_ids.append(self.df_queries.loc[i]["id_left"])
            queries.append(self.df_queries.loc[i]["text_left"])
        self.query_offset += size
        return np.asarray(query_ids), np.asarray(queries)

    """
        Returns the documents batch.
    """

    def generate_docs(self, size=256):
        doc_ids = []
        docs = []
        end = min(self.doc_offset + size, self.docs_len)
        self.docs_file.seek(self.doc_offset) 
            
        for _ in range(self.doc_offset, end):
            doc_id, doc = self.docs_file.readline().rstrip().split("\t", 1)
            doc_ids.append(doc_id)
            docs.append(doc)

        self.doc_offset += size
        return np.asarray(doc_ids), np.asarray(docs)

    """
        Generates qrels.
    """

    def generate_qrels(self):
        return read_qrels(self.qrels)

    def finalize(self):
        self.docs_file.close()
