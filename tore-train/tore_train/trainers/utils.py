import os
import torch
import logging

logger = logging.getLogger(__name__)


def move_to_device(data, device, exclude_keys=None):
    """
    Args:
        data: a list, dict, or torch.Tensor
        device: the target torch.device
        exclude_keys: remove unwanted keys
    """
    if exclude_keys is None:
        exclude_keys = []

    # send data to device
    if isinstance(data, list):
        new_data = []
        for item in data:
            if isinstance(item, torch.Tensor):
                new_data.append(item.to(device, non_blocking=True))
            else:
                new_data.append(move_to_device(item, device, exclude_keys))
        data = new_data

    elif isinstance(data, tuple):
        new_data = ()
        for item in data:
            if isinstance(item, torch.Tensor):
                new_data = new_data + (item.to(device, non_blocking=True), )
            else:
                new_data = new_data + (move_to_device(item, device, exclude_keys), )
        data = new_data

    elif isinstance(data, dict):
        new_data = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and all([key not in k for key in exclude_keys]):
                new_data[k] = v.to(device, non_blocking=True)
            else:
                new_data[k] = move_to_device(v, device, exclude_keys)
        data = new_data

    elif isinstance(data, torch.Tensor) or isinstance(data, torch.nn.Module):
        data = data.to(device, non_blocking=True)
    elif isinstance(data, int) or isinstance(data, float):
        data = data
    else:
        # logger.warning(f"{type(data)} cannot be sent to device")
        data = data
    return data