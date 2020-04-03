import numpy as np
from .inverted_index import InvertedIndex
import json

"""
    Building an inverted index:
    0 - [(doc_id, weight), ...]
    1 - [(doc_id, weight), ...]
    2
    .
    .
    .
    300

    Model should already be trained at this step.
    We take each document and evaluate its representation by the trained model.
    Then according to the representation the inverted index is built.
"""


def build_inverted_index(batch_size, model, eval_loader, iidx_file):
    print("Building inverted index started...")
    inverted_index = InvertedIndex(iidx_file)
    docs_len = eval_loader.docs_length()
    offset = 0
    while offset < docs_len:
        doc_ids, docs = eval_loader.generate_docs(size=batch_size)
        repr = model.evaluate_repr(docs).detach().numpy()
        inverted_index.construct(doc_ids, repr)
        offset += batch_size
    inverted_index.flush()
    print("Inverted index is built!")
    return inverted_index


"""
    This function returns a dictionary, where each query has a correspondent
    relative documents with the corresponding score.
"""


def retrieval_score(model, eval_loader, index, batch_size):
    queries_len = eval_loader.queries_length()
    offset = 0
    res = dict()
    while offset < queries_len:
        queries_id, queries = eval_loader.generate_queries(size=batch_size)
        qreprs = model.evaluate_repr(queries).detach().numpy()
        for qrepr, q in zip(qreprs, queries_id):
            res[int(q)] = retrieval_score_for_query(
                qrepr, index
            )  # returns dict({doc_id:val})
        offset += batch_size
    return res


"""
    This function estimates retrieval score for each query
    over all possible documents from the inverted index.
    
    Returns a dictionaty of shape:
    {
        'doc1_id': val,
        'doc2_id': val
    }
"""


def retrieval_score_for_query(query_repr, index):
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


def dump_retrival_score(score, file):
    json_file = json.dumps(score)
    f = open(file, "w")
    f.write(json_file)
    f.close()
