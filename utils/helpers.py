import os
import json


def manage_model_params(args, model):
    model["model_name"] = "-".join(
        [
            "snrm-model",
            "layers",
            "_".join([str(layer) for layer in model["layers"]]),
            "batch",
            str(model["batch_size"]),
            "dropout",
            str(model["drop_prob"]),
            "learn-rate",
            str(model["learning_rate"]),
            "lambda",
            str(model["reg_lambda"]),
        ]
    )

    dir = os.path.normpath(os.path.join(args.models_folder, model["model_name"]))
    if not os.path.exists(dir):
        os.makedirs(dir)
    model["inverted_index"] = os.path.join(dir, args.inverted_index)
    model["retrieval_score"] = os.path.join(dir, args.retrieval_score)
    model["final_metrics"] = os.path.join(dir, args.final_metrics)
    model["model_pth"] = os.path.join(dir, args.model_pth)
    model["model_checkpoint_pth"] = os.path.join(dir, args.model_checkpoint_pth)


def path_exists(path):
    return os.path.exists(path)


def dump(dict_to_store, filename):
    json_file = json.dumps(dict_to_store)
    f = open(filename, "w")
    f.write(json_file)
    f.close()


def load_file(filename):
    with open(filename, "r") as f:
        loaded_dict = json.load(f)
    return loaded_dict
