import pandas as pd
import numpy as np
from ..pytrec_evaluator import read_qrels
from .helper import load_docs

"""
    EvaluationLoader is used to generate docs and queries batches
    and to get the relevant docs by query.
"""


class EvaluationLoader:
    def __init__(
        self, docs=None, queries=None, docs_dict=None, qrels=None, df_qrels = None,df_queries=None, rs=0
    ):
        self.doc_offset = 0
        self.query_offset = 0
        self.qrels = qrels

        self.__init_df(docs, docs_dict, queries, df_queries, qrels, df_qrels)

    def __build_docs_dict(self, docs):
        self.docs_dict, self.docs_len = load_docs(docs)

    def __init_df(self, docs, docs_dict, queries, df_queries):
        if docs_dict is None:
            self.__build_docs_dict(docs)
        else:
            self.docs_dict = docs_dict
            self.docs_len = len(self.docs_dict)

        if df_qrels is None:
            self.df_qrels = pd.read_csv(self.qrels, sep=" ", na_filter=False, header=None
        )
        else:
            self.df_qrels = df_qrels


        if df_queries is None:
            self.df_queries = pd.read_csv(
                queries, header=None, sep="\t", na_filter=False
            )
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




         batch = []
        qrels_len = df_qrels.shape[0]
        new_offset = min(offset + batch_size, qrels_len)
        is_end = True if new_offset == qrels_len else False
        for i in range(offset, new_offset):
            sample = []
            qrel = df_qrels.loc[i]  # id_query, 0, id_doc
            sample.append(
                df_queries.loc[df_queries["id_left"] == qrel[0]]["text_left"].values[0]
            )
            content = self.__get_content(qrel[2])
            sample.append(content[0])

            if irrelevant:
                sample.append(self.__get_content(content[1])[0])
            offset += 1
            batch.append(sample)
        return batch, is_end, offset

    """
        Returns the documents batch.
    """

    def generate_docs(self, size=256):
        doc_ids = []
        docs = []
        end = min(self.doc_offset + size, self.docs_len)
        self.docs_file.seek(self.doc_offset)

        for _ in range(self.doc_offset, end):
            l = self.docs_file.readline().rstrip().split("\t")
            if len(l) == 2:
                l.append("")
            if len(l) == 1:
                print("Single: ", l, flush=True)
            doc_id, _, doc = l
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
