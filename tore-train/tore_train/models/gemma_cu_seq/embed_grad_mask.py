import torch

def apply_gemma_embed_grad_mask(model):
    """
    Applies a gradient mask to the embedding layer of the given model.

    Args:
        model (torch.nn.Module): The model to apply the gradient mask to.

    Returns:
        torch.nn.Module: The model with the gradient mask applied to the embedding layer.
    """
    def gemma_embedding_backward_hook(grad, cutoff=110):
        """
        Applies a mask to the gradient tensor to zero out the values up to a specified cutoff.

        Args:
            grad (torch.Tensor): The gradient tensor.
            cutoff (int, optional): The cutoff value. Values up to this cutoff will be set to 0. Default is 128000.

        Returns:
            torch.Tensor: The masked gradient tensor.
        """
        grad[cutoff:] = 0
        return grad

    model.model.embed_tokens.weight.register_hook(gemma_embedding_backward_hook)
    return model
