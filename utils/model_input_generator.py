import pandas as pd
import random
from sklearn.model_selection import train_test_split
import numpy as np

"""
All available train and validation data is expected as an input.
"""


class ModelInputGenerator:
    def __init__(self, docs, queries, qrels, valid_size=0.0, rs=0):
        random.seed(0)
        self.docs = docs
        self.queries = queries
        self.qrels = qrels
        self.train_offset = 0
        self.valid_offset = 0
        self.doc_offset = 0
        self.query_offset = 0
        self.__preprocess_data(valid_size, rs)

    def __preprocess_data(self, valid_size, rs):
        print("Preprocessing data started...")

        self.df_docs = pd.read_csv(self.docs, na_filter=False)
        self.df_queries = pd.read_csv(self.queries, na_filter=False)
        df_qrels = pd.read_csv(self.qrels, sep="\t", na_filter=False, header=None)
        if valid_size == 0.0:
            self.df_qrels_train = df_qrels
        else:
            self.df_qrels_train, self.df_qrels_val, _, _ = train_test_split(
                df_qrels, df_qrels, test_size=valid_size, random_state=rs
            )

            self.df_qrels_train = self.df_qrels_train.reset_index()
            self.df_qrels_val = self.df_qrels_val.reset_index()
        self.docs_len = self.df_docs.shape[0]
        self.queries_len = self.df_queries.shape[0]
        print(self.df_qrels_train)
        print("Finished.")

    def __randidx(self, a, b, val):
        while True:
            x = random.randint(a, b)
            if x != val:
                return x

    """
    Returns the batch of 3-values: 'query doc_1(relevant) doc_2(irrelevant)'
    """

    def __generate_batch(self, size, df_qrels, offset):
        batch = []
        qrels_len = df_qrels.shape[0]
        new_offset = min(offset + size, qrels_len)
        is_end = True if new_offset == qrels_len else False
        for i in range(offset, new_offset):
            sample = []
            qrel = df_qrels.loc[i]  # id_query, 0, id_doc
            sample.append(
                self.df_queries.loc[self.df_queries["id_left"] == qrel[0]][
                    "text_left"
                ].values[0]
            )

            sample.append(
                self.df_docs.loc[self.df_docs["id_right"] == qrel[2]][
                    "text_right"
                ].values[0]
            )

            sample.append(
                self.df_docs.loc[self.__randidx(0, self.docs_len - 1, qrel[2])][
                    "text_right"
                ]
            )
            offset += 1
            batch.append(sample)
        return batch, is_end, offset

    def generate_valid_batch(self, size=128):
        batch, is_end, new_off = self.__generate_batch(
            size, self.df_qrels_val, self.valid_offset
        )
        self.valid_offset = new_off
        return batch, is_end

    def generate_train_batch(self, size=256):
        batch, is_end, new_off = self.__generate_batch(
            size, self.df_qrels_train, self.train_offset
        )
        self.train_offset = new_off
        return batch, is_end

    """
        Returns the documents batch
    """

    def generate_docs(self, size=256):
        doc_ids = []
        docs = []
        for i in range(self.doc_offset, self.doc_offset + size):
            doc_ids.append(self.df_docs.loc[i]["id_right"])
            docs.append(self.df_docs.loc[i]["text_right"])
        self.doc_offset += size
        return np.asarray(doc_ids), np.asarray(docs)

    def reset_doc(self, val=0):
        self.doc_offset = val

    def reset(self, val=0):
        self.train_offset = val
        self.valid_offset = val

    def docs_length(self):
        return self.docs_len

    def queries_length(self):
        return self.queries_len

    def generate_queries(self, size=256):
        queries = []
        for i in range(self.query_offset, self.query_offset + size):
            queries.append(self.df_queries.loc[i]["text_left"])
        self.query_offset += size
        return np.asarray(queries)
