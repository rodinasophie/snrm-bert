import pandas as pd
import random
from sklearn.model_selection import train_test_split

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
        self.__preprocess_data(valid_size, rs)

    def __preprocess_data(self, valid_size, rs):
        print("Preprocessing data started...")

        self.df_docs = pd.read_csv(self.docs, na_filter=False)
        self.df_queries = pd.read_csv(self.queries, na_filter=False)
        df_qrels = pd.read_csv(self.qrels, sep="\t", na_filter=False, header=None)
        self.df_qrels_train, self.df_qrels_val, _, _ = train_test_split(
            df_qrels, df_qrels, test_size=valid_size, random_state=rs
        )
        
        self.df_qrels_train = self.df_qrels_train.reset_index()
        self.df_qrels_val = self.df_qrels_val.reset_index()
        self.docs_len = self.df_docs.shape[0]
        print("Finished.")

    def __randidx(self, a, b, val):
        while True:
            x = random.randint(a, b)
            if x != val:
                return x

    """
    Returns the batch of 3-values: 'query doc_1(relevant) doc_2(irrelevant)'
    """

    def __generate_batch(self, size, dftype):
        if dftype == "train":
            df_qrels = self.df_qrels_train
            offset = self.train_offset
        else:
            df_qrels = self.df_qrels_val
            offset = self.valid_offset
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
                self.df_docs.loc[self.__randidx(0, self.docs_len, qrel[2])][
                    "text_right"
                ]
            )
            offset += 1

            batch.append(sample)
        return batch, is_end

    def generate_valid_batch(self, size=128):
        return self.__generate_batch(size, dftype="valid")

    def generate_train_batch(self, size=256):
        return self.__generate_batch(size, dftype = "train")

    """
        Returns the documents batch
    """

    def generate_docs(self, size=256):
        doc_ids = []
        docs = []
        for i in range(self.doc_offset, self.doc_offset + size):
            doc_ids.append(self.df_docs.loc[i]["id_right"])
            docs.append(self.df_docs.loc[i]["text_right"])
        self.offset += size
        return doc_ids, docs

    def reset_doc(self, val=0):
        self.doc_offset = val

    def reset(self, val=0):
        self.train_offset = val
        self.valid_offset = val

    def docs_length(self):
        return self.docs_len
