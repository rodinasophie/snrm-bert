import argparse

from snrm import SNRM
import json
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils.helpers import manage_model_params, path_exists, filename
from utils.evaluation_helpers import evaluate_model
from utils.sparsity import check_sparsity

"""
    Train the model during one epoch over the whole train set.
    Overall training loss is returned.
"""


def train(model, train_loader, batch_size):
    start = datetime.now()
    counter = 0
    while True:
        train_batch, is_end = train_loader.generate_triple_batch(batch_size)
        _ = model.train(train_batch)
        if counter % 1000 == 0:
            print("Train loss: ", model.get_loss("train"), flush=True)
        counter += 1
        if is_end:
            break
    time = datetime.now() - start
    print("Training time: {}".format(time), flush=True)

    return model.get_loss("train")


"""
    Validate the model after full epoch.
    Overall validation loss is returned together with the validation metric.
"""


def validate(model_params, model, valid_loader, batch_size):
    start = datetime.now()
    counter = 0
    while True:
        validation_batch, is_end = valid_loader.generate_triple_batch(batch_size)
        _ = model.validate(validation_batch)
        if counter % 1000 == 0:
            print("Valid loss: ", model.get_loss("valid"), flush=True)
        counter += 1
        if is_end:
            break
    time = datetime.now() - start
    print("Validation time: {}".format(time), flush=True)

    return model.get_loss("valid")


"""
    Saves the model with the minimal given parameter.
"""


def save_model_by_param(
    model_pth, model, new_param, best_param, param_name, e, eval_func
):
    if best_param is None:
        print(
            "Initial model save for epoch: {}, {} = {}".format(e, param_name, new_param)
        )
        best_param = new_param
        model.save(model_pth)

    best_eval = eval_func(new_param, best_param)
    if new_param == best_eval:
        print(
            "Better model is found: {} = {}, epoch = {}".format(
                new_param, param_name, e
            )
        )
        best_param = new_param
        model.save(model_pth)
    return best_param


"""
    Save checkpoint for the case when training was interrupted by external error.
    It's needed to start training again from the checkpoint instead of the very beginning.
"""


def save_checkpoint(model, path, epoch):
    print("Saving checkpoint for epoch #", epoch)
    fname, ext = filename(path)
    model.save_checkpoint(fname + "_epoch" + str(epoch) + ext, epoch)
    model.save_checkpoint(path, epoch)


"""
    Resume model from the last checkpoint if exists.
"""


def resume_model(model, checkpoint_pth, rerun_if_exists, epochs_num):
    if not path_exists(checkpoint_pth):
        print("Model will be trained from scratch.")
        return True, 0

    epoch, is_resumed = model.load_checkpoint(checkpoint_pth, epochs_num)

    if not is_resumed and not rerun_if_exists:
        print("Model exists and retraining is turned off, return old model.")
        return False, _

    print("Model resumed from epoch #", epoch)
    return True, epoch


"""
Training and validating the model.
"""


def train_and_validate(args, model, model_params, train_loader, valid_loader):
    batch_size = model_params["batch_size"]
    is_resumed, start_epoch = resume_model(
        model, model_params["model_checkpoint_pth"], args.rerun_if_exists, args.epochs
    )
    if not is_resumed:
        return
    epochs = range(start_epoch, args.epochs)
    writer = SummaryWriter(args.summary_folder)

    for e in epochs:
        start = datetime.now()
        print("Training, epoch #", e)
        train_loss = train(model, train_loader, batch_size)
        valid_loss = validate(model_params, model, valid_loader, batch_size)

        writer.add_scalars(
            model_params["model_name"],
            {"Validation loss": valid_loss, "Training loss": train_loss},
            e,
        )

        save_checkpoint(model, model_params["model_checkpoint_pth"], e)
        model.save(model_params["model_pth"])
        print("Checking sparsity for epoch #", e, flush=True)
        qmean, dmean = check_sparsity(model, valid_loader, batch_size)
        print(
            "Mean sparsity for epoch {}: queries sparsity = {}, docs sparsity = {}".format(
                e, qmean, dmean
            ),
            flush=True,
        )
        if dmean > 0.95 * model_params["layers"][-1] and e % 5 == 0:
            start1 = datetime.now()
            metrics = evaluate_model(
                model_params, model, valid_loader, args.test_metrics, dump=False
            )
            time1 = datetime.now() - start1
            print("Evaluation time: {}".format(time1), flush=True)
            print("IR metrics: ", metrics)

        model.reset_loss("train")
        model.reset_loss("valid")

        time = datetime.now() - start
        print("Execution time: ", time)

    writer.close()


def run(args, model_params):
    print("\nRunning training for {} ...".format(model_params["model_name"]))
    model = SNRM(
        learning_rate=model_params["learning_rate"],
        batch_size=model_params["batch_size"],
        layers=model_params["layers"],
        reg_lambda=model_params["reg_lambda"],
        drop_prob=model_params["drop_prob"],
        fembeddings=args.embeddings,
        fwords=args.words,
        qmax_len=args.qmax_len,
        dmax_len=args.dmax_len,
        is_stub=args.is_stub,
    )

    docs_dict = dataset.data_loader.load_docs(args.docs)

    train_loader = dataset.data_loader.DataLoader(
        args.train_queries, args.train_qrels, docs_dict,
    )

    valid_loader = dataset.data_loader.DataLoader(
        args.valid_queries, args.valid_qrels, docs_dict,
    )

    train_and_validate(args, model, model_params, train_loader, valid_loader)
    print("Finished training and validating for {}".format(model_params["model_name"]))


def setup(module):
    global dataset
    dataset = __import__(module, fromlist=["object"])


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

    setup(".".join(["utils", args.dataset]))
    models_to_train = list(args.models)
    for model in models_to_train:
        manage_model_params(args, model)
        run(args, model)
