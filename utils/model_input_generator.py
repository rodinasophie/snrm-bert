import pandas as pd


class ModelInputGenerator:
    def __init__(self, docs, queries, qrels):
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
        print("Finished.")

    """
    Returns the batch of 4-values: 'query doc_1 doc_2 relevance_label'
    """

    def generate_batch(self, size=256):
        batch = []
        for i in range(self.offset, self.offset + size):
            sample = []
            qrel = self.df_qrels.loc[i]  # id_query, 0, id_doc, relevance
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
                self.df_docs.loc[self.df_docs["id_right"] != qrel[2]][
                    "text_right"
                ].values[0]
            )
            sample.append("1")

            batch.append(sample)
        self.offset += size
        return batch
