import os
import pickle
import torch


# select CPU or GPU as compute device
def choose_device(args):
    device_name = args.device
    if device_name == "cpu":
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device(device_name)
    else:
        raise Exception("CUDA not available")
    return device


# check for required folders
def check_paths(args):
    assert os.path.isdir(args.input_path)
    cache_path = args.cache_path
    output_path = args.output_path
    graph_cache = f"{cache_path}/graphs"
    dirs = [cache_path, output_path, graph_cache]
    for dir in dirs:
        if not os.path.isdir(dir):
            os.mkdir(dir)


def cached(f, cache_file, args):
    # unpack args
    if type(args) == type({}):
        cache_path = args["cache_path"]
        recache = args["recache"]
    else:
        cache_path = args.cache_path
        recache = args.recache
    # determine full cache file path
    cache_file = f"{cache_path}/{cache_file}"
    if recache or not os.path.isfile(cache_file):
        # run f() and cache result
        output = f()
        with open(cache_file, "wb") as cf:
            pickle.dump(output, cf)
    else:
        # load cached result
        with open(cache_file, "rb") as cf:
            output = pickle.load(cf)
    return output ## TODO hack to `rm -rf cache`


def save_model(model, args):
    with open(f"{args.output_path}/trained_model.pkl", "wb") as f:
        pickle.dump(model, f)


def save_checkpoint(model, args):
    with open(f"{args.cache_path}/checkpoint.pkl", "wb") as f:
        pickle.dump(model, f)


def print_args(args):
    print("\nARGS")
    print("########################################")
    print(args_string(args))
    print("########################################\n")


def args_string(args):
    argstr = ""
    arg_dict = vars(args)
    for key in arg_dict:
        argstr += f"{key}: {arg_dict[key]}\n"
    return argstr


def write_args(args):
    output_path = args.output_path
    with open(f"{output_path}/args.txt", "w") as f:
        f.write(args_string(args))
