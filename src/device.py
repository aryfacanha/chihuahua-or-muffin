import torch


def get_device(prefer_cuda=True):
    if not prefer_cuda or not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        torch.empty(1, device="cuda")
    except (AssertionError, RuntimeError):
        return torch.device("cpu")

    return torch.device("cuda")


def describe_device(device):
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
        return f"cuda ({device_name})"

    return "cpu"
