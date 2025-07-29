import torch

def rearrange(tensor: torch.Tensor, pattern: str) -> torch.Tensor:
    """
    Rearranges the tensor dimensions according to a simple pattern.

    Currently only supports permute patterns like: "b c h -> b h c"

    Args:
        tensor (Tensor): Input tensor.
        pattern (str): A string like "b c h -> b h c".

    Returns:
        Tensor: Rearranged tensor.
    """
    try:
        left, right = pattern.split("->")
        left = left.strip().split()
        right = right.strip().split()
    except Exception:
        raise ValueError(f"Invalid pattern format: '{pattern}'")

    if sorted(left) != sorted(right):
        raise ValueError("Pattern must be a permutation of input dims.")

    dim_map = {dim: i for i, dim in enumerate(left)}
    permute_dims = [dim_map[dim] for dim in right]

    return tensor.permute(*permute_dims)
