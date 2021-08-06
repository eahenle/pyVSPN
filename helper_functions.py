import torch
import os
import pickle

# select CPU or GPU as compute device
def choose_device(args):
    device_name = args.device
    if device_name == "cpu":
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device(device_name)
    else:
        raise Exception("CUDA not available")
    return device, torch.device("cpu")


# check for required folders
def check_paths(args):
    if not os.path.isdir(args.cache_path):
        os.mkdir(args.cache_path)


def cached(f, cache_file, args):
    # unpack args
    cache_path = args.cache_path
    recache = args.recache
    # determine full cache file path
    cache_file = f"{cache_path}/{cache_file}"
    if recache or not os.path.isfile(cache_file):
        # run f(farg) and cache result
        output = f()
        with open(cache_file, "wb") as cf:
            pickle.dump(output, cf)
    else:
        # load cached result
        with open(cache_file, "rb") as cf:
            output = pickle.load(cf)
    return output


def save_model(model, args):
    with open(f"{args.model_output}", "wb") as f:
        pickle.dump(model, f)


def save_checkpoint(model, args):
    cached(lambda : model, "model_checkpoint.pkl", args)
