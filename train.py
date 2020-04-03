import argparse

from snrm import SNRM, InvertedIndex
from utils import ModelInputGenerator
import json
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from snrm.inverted_index import build_inverted_index


def train(model, data_generator, batch_size):
    while True:
        train_batch, is_end = data_generator.generate_train_batch(size=batch_size)
        _ = model.train(train_batch)
        if is_end:
            break
    return model.get_loss("train")


def validate(model, data_generator, batch_size):
    while True:
        validation_batch, is_end = data_generator.generate_valid_batch(size=batch_size)
        _ = model.validate(validation_batch)
        if is_end:
            break
    return model.get_loss("valid")


"""
Training and validating the model.
"""


def train_and_validate(args, model, data_generator):
    writer = SummaryWriter(args.summary_folder)

    batch_size = args.batch_size
    epochs = args.epochs
    for e in range(epochs):
        start = datetime.now()
        print("Training, epoch #", e)
        data_generator.reset()
        train_loss = train(model, data_generator, batch_size)
        valid_loss = validate(model, data_generator, batch_size)
        writer.add_scalars(
            "snrm-run-1",
            {"Training loss": train_loss, "Validation loss": valid_loss},
            e,
        )
        model.reset_loss("train")
        model.reset_loss("valid")
        time = datetime.now() - start
        print("Execution time: ", time)
        print("Train loss: ", train_loss)
        print("Valid loss: ", valid_loss)

    writer.close()


def run(args):
    print("Running....")
    model = SNRM(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        layers=args.layers,
        reg_lambda=args.reg_lambda,
        drop_prob=args.drop_prob,
        fembeddings=args.embeddings,
        qmax_len=args.qmax_len,
        dmax_len=args.dmax_len,
        is_stub=args.is_stub,
    )
    mi_generator = ModelInputGenerator(
        args.docs, args.queries, args.qrels, valid_size=args.valid_size
    )

    train_and_validate(args, model, mi_generator)
    # FIXME: inverted index is not needed on train stage, it's needed on testing stage
    # build_inverted_index(args.batch_size, model, mi_generator, args.inverted_index)
    model.save(args.model)


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
