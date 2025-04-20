import torch

def zigzag_ring_attn_split(
    value: torch.Tensor,
    rank: int,
    world_size: int,
    device: torch.device = None,
    dim: int = 1,
) -> torch.Tensor:
    """
    Splits the input tensor into local chunks for zigzag ring attention.

    Args:
        value (torch.Tensor): The input tensor to be split, with shape (batch_size, seq_len, ...).
        rank (int): The rank of the current process in the distributed setup.
        world_size (int): The total number of processes in the distributed setup.
        device (torch.device, optional): The device to which the local chunks should be moved. Defaults to None.
        dim (int, optional): The dimension along which to split the tensor. Defaults to 1.

    Returns:
        torch.Tensor: The concatenated local chunks for the current process.
    """
    value_chunks = value.chunk(2 * world_size, dim=dim)
    local_value = torch.cat(
        [value_chunks[rank], value_chunks[2 * world_size - rank - 1]], dim=dim
    )
    if device is not None:
        local_value = local_value.to(device)
    return local_value
