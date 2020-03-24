import argparse

from snrm import SNRM, InvertedIndex
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
    print("Building inverted index started...")
    inverted_index = InvertedIndex(iidx_file)
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
Training and validating the model.
"""


def train_and_validate(args, model, data_generator):
    writer = SummaryWriter(args.summary_folder)

    batch_size = args.batch_size
    epochs = args.epochs
    for e in range(epochs):
        print("Training, epoch #", e)
        data_generator.reset()

        while True:
            train_batch, is_end = data_generator.generate_train_batch(size=batch_size)
            _ = model.train(train_batch)

            if is_end:
                break

        while True:
            validation_batch, is_end = data_generator.generate_valid_batch(
                size=batch_size
            )
            _ = model.validate(validation_batch)
            if is_end:
                break

        writer.add_scalars(
            "snrm-run-1",
            {
                "Training loss": model.get_loss("train"),
                "Validation loss": model.get_loss("valid"),
            },
            e,
        )
        model.reset_loss("train")
        model.reset_loss("valid")

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
    mi_generator = ModelInputGenerator(
        args.docs, args.queries, args.qrels, valid_size=0.2
    )

    train_and_validate(args, model, mi_generator)
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
