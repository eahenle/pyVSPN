from genericpath import isdir
import torch
import os

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
    if not os.path.isdir(args.cache_path):
        os.mkdir(args.cache_path)
