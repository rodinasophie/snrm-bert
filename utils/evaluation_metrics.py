import numpy as np


def retrieval_score(query, index):
    relevant_docs = dict()
    for i in range(len(query)):
        if query[i] != 0.0:
            docs = index.get_index()[i]
            for j in range(len(docs)):
                doc_id = docs[j][0]
                if doc_id not in relevant_docs:
                    relevant_docs[doc_id] = query[i] * docs[j][1]
                else:
                    relevant_docs[doc_id] += query[i] * docs[j][1]
    maximum = max(relevant_docs, key=relevant_docs.get)
    print("Max key and value: ", maximum, relevant_docs[maximum])
    return maximum
