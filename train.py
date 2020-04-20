import argparse

from snrm import SNRM
import json
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils.helpers import manage_model_params, path_exists
from utils.evaluation_helpers import evaluate_model


"""
    Train the model during one epoch over the whole train set.
    Overall training loss is returned.
"""


def train(model, train_loader, batch_size):
    start = datetime.now()
    while True:
        train_batch, is_end = train_loader.generate_train_batch(batch_size)
        _ = model.train(train_batch)
        if is_end:
            break
    time = datetime.now() - start
    print("Training time: {}".format(time), flush=True)
    return model.get_loss("train")


"""
    Validate the model after full epoch.
    Overall validation loss is returned together with the validation metric.
"""


def validate(
    model_params, model, train_loader, batch_size, docs_filename, valid_metric=None
):
    start = datetime.now()
    while True:
        validation_batch, is_end = train_loader.generate_valid_batch(
            batch_size, irrelevant=True, force_keep=True
        )
        _ = model.validate(validation_batch)
        if is_end:
            break
    time = datetime.now() - start
    print("Validation[1] time: {}".format(time), flush=True)

    start = datetime.now()
    metric = None
    if valid_metric is not None:
        eval_loader = dataset.evaluation_loader.EvaluationLoader(
            docs=docs_filename,
            df_queries=train_loader.get_valid_queries_ref(),
            qrels=train_loader.get_valid_qrels_name(),
        )
        metric = evaluate_model(
            model_params, model, eval_loader, [valid_metric], dump=False
        )
    train_loader.unload_all()
    time = datetime.now() - start
    print("Validation[2] time: {}".format(time), flush=True)
    return model.get_loss("valid"), metric


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
    if new_param == best_eval and best_eval != best_param:
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


def train_and_validate(args, model, model_params, train_loader):
    batch_size = model_params["batch_size"]
    is_resumed, start_epoch = resume_model(
        model, model_params["model_checkpoint_pth"], args.rerun_if_exists, args.epochs
    )
    if not is_resumed:
        return
    epochs = range(start_epoch, args.epochs)
    writer = SummaryWriter(args.summary_folder)

    best_metric = None
    for e in epochs:
        start = datetime.now()
        print("Training, epoch #", e)
        train_loss = train(model, train_loader, batch_size)
        save_checkpoint(model, model_params["model_checkpoint_pth"], e)
        
        valid_loss, valid_metric = validate(
            model_params, model, train_loader, batch_size, args.docs, args.valid_metric
        )
        
        best_metric = save_model_by_param(
            model_params["model_pth"],
            model,
            valid_metric[args.valid_metric],
            best_metric,
            "IR metric: {}".format(args.valid_metric),
            e,
            max,
        )
        writer.add_scalars(
            model_params["model_name"],
            {
                "Training loss": train_loss,
                "Validation loss": valid_loss,
                "IR metric({})".format(args.valid_metric): valid_metric[
                    args.valid_metric
                ],
            },
            e,
        )
        

        model.reset_loss("train")
        model.reset_loss("valid")
        time = datetime.now() - start
        print("Execution time: ", time)
        print("Train loss: ", train_loss)
        print("Valid loss: ", valid_loss)
        print("IR metric: ", valid_metric)

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
        fwords = args.words,
        qmax_len=args.qmax_len,
        dmax_len=args.dmax_len,
        is_stub=args.is_stub,
    )
    train_loader = dataset.train_loader.TrainLoader(
        args.docs,
        args.train_queries,
        args.train_qrels,
        args.valid_queries,
        args.valid_qrels,
        save_mem=args.save_mem,
    )

    train_and_validate(args, model, model_params, train_loader)
    train_loader.finalize()
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
