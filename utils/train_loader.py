import pandas as pd
import random
import numpy as np

"""
    TrainLoader is responsible for triples generation
    for the train cycle.

    Train and validation sets are not kept in memory concurrently
    for memory efficiency purposes.
"""


class TrainLoader:
    def __init__(self, docs, train_queries, train_qrels, valid_queries, valid_qrels):
        self.train_queries = train_queries
        self.train_qrels = train_qrels

        self.valid_queries = valid_queries
        self.valid_qrels = valid_qrels

        self.is_trainset_loaded = False
        self.is_validset_loaded = False

        self.__load_docs(docs)

    """
        Loads the documents into the memory.
    """

    def __load_docs(self, docs):
        self.df_docs = pd.read_csv(docs, na_filter=False)
        self.docs_len = self.df_docs.shape[0]

    """
        Loads train queries and qrels into the memory.
    """

    def __load_trainset(self):
        self.train_offset = 0
        self.df_train_queries = pd.read_csv(self.train_queries, na_filter=False)
        self.df_train_qrels = pd.read_csv(
            self.train_qrels, sep="\t", na_filter=False, header=None
        )
        self.is_trainset_loaded = True
        print("Training set is loaded to memory.")

    """
        Loads validation queries and qrels into the memory.
    """

    def __load_validset(self):
        self.valid_offset = 0
        self.df_valid_queries = pd.read_csv(self.valid_queries, na_filter=False)
        self.df_valid_qrels = pd.read_csv(
            self.valid_qrels, sep="\t", na_filter=False, header=None
        )
        self.is_validset_loaded = True
        print("Validation set is loaded to memory.")

    """
        Unload trainset dataframes and release memory.
    """

    def __unload_trainset(self):
        self.train_offset = 0
        del self.df_train_queries
        del self.df_train_qrels
        self.is_trainset_loaded = False
        print("Train set is released.")

    """
        Unload validset dataframes and release memory.
    """

    def __unload_validset(self):
        self.valid_offset = 0
        del self.df_valid_queries
        del self.df_valid_qrels
        self.is_validset_loaded = False
        print("Validation set is released.")

    """
        Generates a random number from (a, b) except val.
    """

    def __randidx(self, a, b, val):
        while True:
            x = random.randint(a, b)
            if x != val:
                return x

    """
        General function for batch generation.
    """

    def __generate_batch(self, df_queries, df_qrels, offset, batch_size):
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

    """
        Generates a triple (query, relevant doc, irrelevant doc) from the train set.
    """

    def generate_train_batch(self, batch_size=128):
        if not self.is_trainset_loaded:
            self.__load_trainset()

        batch, is_end, new_off = self.__generate_batch(
            self.df_train_queries, self.df_train_qrels, self.train_offset, batch_size
        )
        self.train_offset = new_off
        if is_end:
            self.__unload_trainset()
        return batch, is_end

    """
        Generates a triple (query, relevant doc, irrelevant doc) from the valid set.
    """

    def generate_valid_batch(self, batch_size=128):
        if not self.is_validset_loaded:
            self.__load_validset()

        batch, is_end, new_off = self.__generate_batch(
            self.df_valid_queries, self.df_valid_qrels, self.valid_offset, batch_size
        )

        self.valid_offset = new_off
        if is_end:
            self.__unload_validset()
        return batch, is_end
