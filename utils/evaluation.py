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
