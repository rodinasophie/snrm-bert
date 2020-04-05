import pandas as pd
import random
import numpy as np

"""
    EvaluationLoader is used to generate docs and queries batches
    and to get the relevant docs by query.
"""


class EvaluationLoader:
    def __init__(self, docs, queries, rs=0):
        random.seed(rs)
        self.doc_offset = 0
        self.query_offset = 0
        self.__init_df(docs, queries)

    def __init_df(self, docs, queries):
        self.df_docs = pd.read_csv(docs, na_filter=False)
        self.df_queries = pd.read_csv(queries, na_filter=False)

        self.docs_len = self.df_docs.shape[0]
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
        for i in range(self.doc_offset, end):
            doc_ids.append(self.df_docs.loc[i]["id_right"])
            docs.append(self.df_docs.loc[i]["text_right"])
        self.doc_offset += size
        return np.asarray(doc_ids), np.asarray(docs)

    def get_docs_by_query(self, q):
        # TODO: q is query
        pass
