import pandas as pd
import numpy as np
from ..pytrec_evaluator import read_qrels


def load_docs(docs):
    docs_file = open(docs, "rt", encoding="utf8")
    docs_dict = dict()

    for line in docs_file:
        l = line.rstrip().split("\t")
        doc_id, irr_doc_id, doc = l
        docs_dict[doc_id] = (doc, irr_doc_id)

    docs_file.close()
    print("Documents are loaded in data_loader")
    return docs_dict


class DataLoader:
    def __init__(self, queries, qrels, docs_dict):
        self.qrels = qrels
        self.queries = queries
        self.docs_dict = docs_dict

        self.__load_data()

        self.triple_offset = 0
        self.query_offset = 0

        self.doc_ids_generated = False
        self.local_doc_ids = None
        self.local_docs_offset = 0

    def __load_data(self):
        self.df_queries = pd.read_csv(
            self.queries,
            sep="\t",
            names=["id_left", "text_left"],
            na_filter=False,
            header=None,
        )
        self.queries_len = self.df_queries.shape[0]

        self.df_qrels = pd.read_csv(self.qrels, sep=' ', index_col = False, na_filter=False, header=None)
        self.qrels_len = self.df_qrels.shape[0]

    def __get_content(self, doc_id):
        return self.docs_dict[doc_id]

    def generate_triple_batch(self, batch_size):
        queries = []
        docs_1 = []
        docs_2 = []
        qrels_len = self.df_qrels.shape[0]
        new_offset = min(self.triple_offset + batch_size, qrels_len)
        is_end = True if new_offset == qrels_len else False

        for i in range(self.triple_offset, new_offset):
            qrel = self.df_qrels.loc[i]  # id_query, 0, id_doc
            queries.append(
                self.df_queries.loc[self.df_queries["id_left"] == qrel[0]][
                    "text_left"
                ].values[0]
            )
            content = self.__get_content(str(qrel[2]))
            docs_1.append(content[0])

            docs_2.append(self.__get_content(str(content[1]))[0])
        self.triple_offset = new_offset

        if is_end:
            self.triple_offset = 0
        return [queries, docs_1, docs_2], is_end

    """
        Returns the query batch.
    """

    def generate_queries(self, batch_size):
        query_ids = []
        queries = []
        end = min(self.query_offset + batch_size, self.queries_len)
        for i in range(self.query_offset, end):
            query_ids.append(self.df_queries.loc[i]["id_left"])
            queries.append(self.df_queries.loc[i]["text_left"])
        self.query_offset = end
        is_end = False

        if self.query_offset == self.queries_len:
            is_end = True
            self.query_offset = 0
        return np.asarray(query_ids), np.asarray(queries), is_end

    def __generate_unique_doc_ids(self):
        s = set()
        for i in range(self.qrels_len):
            s.add(self.df_qrels.loc[i][2])
        return list(s)

    """
        Returns the documents batch.
    """

    def generate_docs(self, batch_size):
        if not self.doc_ids_generated:
            self.local_doc_ids = self.__generate_unique_doc_ids()
            self.doc_ids_generated = True

        local_docs_len = len(self.local_doc_ids)
        end = min(self.local_docs_offset + batch_size, local_docs_len)

        docs = []
        for i in range(self.local_docs_offset, end):
            docs.append(self.docs_dict[str(self.local_doc_ids[i])][0])
        doc_ids = self.local_doc_ids[self.local_docs_offset : end]

        self.local_docs_offset = end

        is_end = False
        if end == local_docs_len:
            is_end = True
            self.local_docs_offset = 0
        return np.asarray(doc_ids), np.asarray(docs), is_end

    """
        Generates qrels.
    """

    def generate_qrels(self):
        return read_qrels(self.qrels)
    
    def reset(self):
        self.local_docs_offset = 0
        self.query_offset = 0
        self.triple_offset = 0


        
