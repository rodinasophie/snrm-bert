import argparse

from snrm import SNRM
from utils import TrainLoader
import json
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils.manage_model import manage_model_params

"""
    Train the model during one epoch over the whole train set.
    Overall training loss is returned.
"""


def train(model, train_loader, batch_size):
    while True:
        train_batch, is_end = train_loader.generate_train_batch(batch_size)
        _ = model.train(train_batch)
        if is_end:
            break
    return model.get_loss("train")


"""
    Validate the model after full epoch.
    Overall validation loss is returned.
"""


def validate(model, train_loader, batch_size):
    while True:
        validation_batch, is_end = train_loader.generate_valid_batch(batch_size)
        _ = model.validate(validation_batch)
        if is_end:
            break
    return model.get_loss("valid")


"""
    Saves the model with the minimal validation loss.
"""


def save_model_by_validloss(model_pth, model, valid_loss, min_loss, e):
    if min_loss is None:
        print("Initial model save for epoch: {}, valid_loss = {}".format(e, valid_loss))
        min_loss = valid_loss
        model.save(model_pth)

    if valid_loss < min_loss:
        print(
            "Better model is found: valid_loss = {}, epoch = {}".format(valid_loss, e)
        )
        min_loss = valid_loss
        model.save(model_pth)
    return min_loss


def save_model_by_eval(model, train_loader, batch_size):
    pass


"""
Training and validating the model.
"""


def train_and_validate(args, model, model_params, train_loader):
    writer = SummaryWriter(args.summary_folder)

    batch_size = model_params["batch_size"]
    epochs = args.epochs
    min_loss = None
    for e in range(epochs):
        start = datetime.now()
        print("Training, epoch #", e)
        train_loss = train(model, train_loader, batch_size)
        valid_loss = validate(model, train_loader, batch_size)
        writer.add_scalars(
            model_params["model_name"],
            {"Training loss": train_loss, "Validation loss": valid_loss},
            e,
        )
        min_loss = save_model_by_validloss(
            model_params["model_pth"], model, valid_loss, min_loss, e
        )
        # TODO: save_model_by_eval?

        model.reset_loss("train")
        model.reset_loss("valid")
        time = datetime.now() - start
        print("Execution time: ", time)
        print("Train loss: ", train_loss)
        print("Valid loss: ", valid_loss)

    writer.close()


def run(args, model_params):
    print("Running....")
    model = SNRM(
        learning_rate=model_params["learning_rate"],
        batch_size=model_params["batch_size"],
        layers=model_params["layers"],
        reg_lambda=model_params["reg_lambda"],
        drop_prob=model_params["drop_prob"],
        fembeddings=args.embeddings,
        qmax_len=args.qmax_len,
        dmax_len=args.dmax_len,
        is_stub=args.is_stub,
    )
    train_loader = TrainLoader(
        args.docs,
        args.train_queries,
        args.train_qrels,
        args.valid_queries,
        args.valid_qrels,
        save_mem=args.save_mem,
    )

    train_and_validate(args, model, model_params, train_loader)


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
    models_to_train = list(args.models)
    for model in models_to_train:
        manage_model_params(args, model)
        run(args, model)
