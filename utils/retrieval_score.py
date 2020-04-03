import json


class RetrievalScore:
    def __init__(self):
        self.retrieval_score = None
        self.is_evaluated = False

    """
        This function returns a dictionary, where each query has a correspondent
        relative documents with the corresponding score.
    """

    def evaluate(self, eval_loader, index, model, batch_size):
        if self.is_evaluated:
            return self.retrieval_score
        queries_len = eval_loader.queries_length()
        offset = 0
        self.retrieval_score = dict()
        while offset < queries_len:
            queries_id, queries = eval_loader.generate_queries(size=batch_size)
            qreprs = model.evaluate_repr(queries).detach().numpy()
            for qrepr, q in zip(qreprs, queries_id):
                self.retrieval_score[int(q)] = self.__retrieval_score_for_query(
                    qrepr, index
                )  # returns dict({doc_id:val})
            offset += batch_size
        self.is_evaluated = True
        return self.retrieval_score

    """
        This function estimates retrieval score for each query
        over all possible documents from the inverted index.
        
        Returns a dictionaty of shape:
        {
            'doc1_id': val,
            'doc2_id': val
        }
    """

    def __retrieval_score_for_query(self, query_repr, index):
        relevant_docs = dict()
        for i in range(len(query_repr)):
            if query_repr[i] != 0.0:
                docs = index.get_index()[i]
                for j in range(len(docs)):
                    doc_id = int(docs[j][0])
                    if doc_id not in relevant_docs:
                        relevant_docs[doc_id] = query_repr[i].item() * docs[j][1]
                    else:
                        relevant_docs[doc_id] += query_repr[i].item() * docs[j][1]
        return relevant_docs

    """
        Dump retrieval score.
    """

    def dump(self, filename):
        if not self.is_evaluated:
            print("Retrieval score is not yet evaluated")
        else:
            json_file = json.dumps(self.retrieval_score)
            f = open(filename, "w")
            f.write(json_file)
            f.close()

    """
        Read retrieval score to dict.
    """

    def read(self, filename):
        if self.is_evaluated:
            print("One retrieval score is already loaded")
        else:
            with open(filename, "r") as f:
                self.retrieval_score = json.load(f)
        return self.retrieval_score
