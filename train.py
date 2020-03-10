import argparse
from model.snrm import SNRM
from utils import ModelInputGenerator

"""
Training the model.
"""


def train_model(args):
    # TODO:
    # 1. Train network with the data
    # 2. Build inverted index from the training data
    # 3. Store the model

    # 4. TODO: add functionality to

    model = SNRM()
    mi_generator = ModelInputGenerator(args.docs, args.queries, args.qrels)
    steps = 2

    while steps != 0:
        batch = mi_generator.generate_batch(size=4)
        print(batch)
        model.train(batch)
        steps -= 1

    model.save(args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=str, help="Path to training documents")
    parser.add_argument("--queries", type=str, help="Path to training queries")
    parser.add_argument("--qrels", type=str, help="Path to training qrels")

    parser.add_argument("--output-file", type=str, help="Path to store output model")

    args = parser.parse_args()
    train_model(args)

