import json


class InvertedIndex:
    def __init__(self, out_file):
        self.index = dict()
        self.out_file = out_file

    def construct(self, doc_ids, doc_repres):
        repres_len = doc_repres.shape[1]
        for i in range(doc_repres.shape[0]):
            for j in range(repres_len):
                if doc_repres[i][j] > 0.0:
                    if j not in self.index:
                        self.index[j] = []
                    self.index[j].append([int(doc_ids[i]), doc_repres[i][j].item()])

    def get_index(self):
        return self.index

    def flush(self):
        json_file = json.dumps(self.index)
        f = open(self.out_file, "w")
        f.write(json_file)
        f.close()

    def read_index(self):
        with open(self.out_file, "r") as f:
            self.index = json.load(f)
        return self.index


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


def build_inverted_index(batch_size, model, mi_generator, iidx_file):
    print("Building inverted index started...")
    inverted_index = InvertedIndex(iidx_file)
    docs_len = mi_generator.docs_length()
    offset = 0
    while offset < docs_len:
        doc_ids, docs = mi_generator.generate_docs(size=batch_size)
        print(docs)
        repr = model.evaluate_repr(docs)
        inverted_index.construct(doc_ids, repr)
        offset += batch_size
    inverted_index.flush()
    print("Inverted index is built!")
    return inverted_index
