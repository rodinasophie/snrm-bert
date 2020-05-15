import pandas as pd
import random
import numpy as np

"""
    TrainLoader is responsible for triples generation
    for the train cycle.

    Train and validation sets are not kept in memory concurrently
    for memory efficiency purposes.

    Params:
        save_mem - if this flag is set, after all data is read(is_end=True)
        it's removed from the memory
"""


class TrainLoader:
    def __init__(
        self, docs, train_queries, train_qrels, valid_queries, valid_qrels, save_mem,
    ):
        self.train_queries = train_queries
        self.train_qrels = train_qrels

        self.valid_queries = valid_queries
        self.valid_qrels = valid_qrels

        self.save_mem = save_mem

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
        if not self.is_trainset_loaded:
            return
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
        if self.df_valid_queries is not None:
            del self.df_valid_queries
        if self.df_valid_qrels is not None:
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

    def __generate_batch(self, df_queries, df_qrels, offset, batch_size, irrelevant):
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
            if irrelevant:
                sample.append(
                    self.df_docs.loc[self.__randidx(0, self.docs_len - 1, qrel[2])][
                        "text_right"
                    ]
                )
            batch.append(sample)
        return batch, is_end, offset

    """
        Generates a triple (query, relevant doc, irrelevant doc) from the train set.
    """

    def generate_train_batch(
        self, batch_size=128,
    ):
        if not self.is_trainset_loaded:
            self.__load_trainset()

        batch, is_end, new_off = self.__generate_batch(
            self.df_train_queries,
            self.df_train_qrels,
            self.train_offset,
            batch_size,
            True,
        )
        self.train_offset = new_off

        if is_end and self.save_mem:
            self.__unload_trainset()
        elif is_end:
            self.train_offset = 0

        return batch, is_end

    """
        Generates a triple (query, relevant doc, irrelevant doc) from the valid set.
    """

    def generate_valid_batch(self, batch_size=128, irrelevant=False, force_keep=False):
        if not self.is_validset_loaded:
            self.__load_validset()

        batch, is_end, new_off = self.__generate_batch(
            self.df_valid_queries,
            self.df_valid_qrels,
            self.valid_offset,
            batch_size,
            irrelevant,
        )

        self.valid_offset = new_off

        if is_end and force_keep:
            self.valid_offset = 0
        elif is_end and self.save_mem:
            self.__unload_validset()
        elif is_end:
            self.valid_offset = 0

        return batch, is_end

    """
        Return docs ref to store memory.
    """

    def get_docs_ref(self):
        return self.df_docs

    """
        Return valid queries ref.
    """

    def get_valid_queries_ref(self):
        assert (
            self.df_valid_queries is not None
        ), "validation queries data frame is None"
        return self.df_valid_queries

    """
        Return qrels name.
    """
    # FIXME: interface is asymmetric because of this function.
    # But we store here qrels in data frame, but we need it in neighborhood module in pytrec_dict.
    def get_valid_qrels_name(self):
        return self.valid_qrels

    """
        Unload all data from the memory.
    """

    def unload_all(self):
        if self.save_mem:
            self.__unload_validset()
            self.__unload_trainset()

    def finalize(self):
        pass
