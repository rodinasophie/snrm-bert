import argparse

from model.snrm import SNRM, InvertedIndexConstructor
from utils import ModelInputGenerator
import json


def build_inverted_index(args, model, mi_generator, iidx_file):
    inverted_index = InvertedIndexConstructor(iidx_file)
    batch_size = args.batch_size
    docs_len = mi_generator.docs_length()
    offset = 0
    while offset < docs_len:
        doc_batch = mi_generator.generate_docs(size=batch_size)

        repr = model.evaluate_repr(doc_batch)

        inverted_index.construct(repr)
        inverted_index.flush()
        offset += batch_size


"""
Training the model.
"""


def train(args, model, mi_generator):
    batch_size = args.batch_size
    epoches = args.epoches
    qrel_len = mi_generator.qrel_length()
    for _ in range(epoches):
        mi_generator.reset()
        offset = 0
        while offset < qrel_len:
            batch = mi_generator.generate_batch(size=batch_size)
            model.train(batch)
            offset += batch_size


def run(args):
    model = SNRM(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        layers=args.layers,
        reg_lamdba=args.reg_lambda,
        drop_prob=args.drop_prob,
        fembeddings=args.embeddings,
        qmax_len=args.qmax_len,
        dmax_len=args.dmax_len,
    )
    mi_generator = ModelInputGenerator(args.docs, args.queries, args.qrels)

    train(args, model, mi_generator)
    build_inverted_index(args, model, mi_generator, args.inverted_index)
    model.save(args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--params", type=str, help="Path to json-file with params"
    )
    args, _ = parser.parse_known_args()
    with open(args.params) as f:
        params = json.load(f)
    complex_list = ["learning_rate", "reg_lambda"]
    for key, val in params.items():
        if key in complex_list:
            parser.add_argument("--" + key, default=val["value"] * val["power"])
        else:
            parser.add_argument("--" + key, default=val)
    args = parser.parse_args()
    run(args)

