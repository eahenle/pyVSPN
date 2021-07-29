import torch

def choose_device(device_name):
    if device_name == "cpu":
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device(device_name)
    else:
        raise Exception("CUDA not available")
    return device
