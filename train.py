import argparse

from snrm import SNRM, InvertedIndexConstructor
from utils import ModelInputGenerator
import json
from torch.utils.tensorboard import SummaryWriter


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


def build_inverted_index(args, model, mi_generator, iidx_file):
    inverted_index = InvertedIndexConstructor(iidx_file)
    batch_size = args.batch_size
    docs_len = mi_generator.docs_length()
    offset = 0
    while offset < docs_len:
        doc_ids, docs = mi_generator.generate_docs(size=batch_size)
        repr = model.evaluate_repr(docs)
        inverted_index.construct(doc_ids, repr)
        inverted_index.flush()
        offset += batch_size


"""
Training the model.
"""


def train(args, model, mi_generator):
    writer = SummaryWriter(args.summary_folder)

    batch_size = args.batch_size
    epochs = args.epochs
    qrel_len = mi_generator.qrel_length()
    for e in range(epochs):
        mi_generator.reset()
        offset = 0
        while offset < qrel_len:
            batch = mi_generator.generate_batch(size=batch_size)
            training_loss, validation_loss = model.train(batch)
            offset += batch_size

            writer.add_scalars(
                "snrm-run-0",
                {"Training loss": training_loss, "Validation loss": validation_loss},
                e,
            )

        # model is trained by the i-th epoch

    writer.close()


def run(args):
    model = SNRM(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        layers=args.layers,
        reg_lambda=args.reg_lambda,
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
    for key, val in params.items():
        parser.add_argument("--" + key, default=val)
    args = parser.parse_args()
    print(args)
    run(args)

