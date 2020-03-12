import pandas as pd
import random


class ModelInputGenerator:
    def __init__(self, docs, queries, qrels):
        random.seed(0)
        self.docs = docs
        self.queries = queries
        self.qrels = qrels
        self.offset = 0
        self.__preprocess_data()

    def __preprocess_data(self):
        print("Preprocessing data started...")
        self.df_docs = pd.read_csv(self.docs, na_filter=False)
        self.df_queries = pd.read_csv(self.queries, na_filter=False)
        self.df_qrels = pd.read_csv(self.qrels, sep="\t", na_filter=False, header=None)
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

    def generate_batch(self, size=256):
        batch = []
        for i in range(self.offset, self.offset + size):
            sample = []
            qrel = self.df_qrels.loc[i]  # id_query, 0, id_doc
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
                self.df_docs.loc[self.__randidx(0, self.docs_len, qrel[2])]["text_right"]
            )

            batch.append(sample)
        self.offset += size
        return batch
    
    def reset(self, val=0):
        self.offset = val
