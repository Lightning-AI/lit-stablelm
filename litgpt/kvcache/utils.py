import torch


def bits_for_torch_dtype(dtype: torch.dtype) -> int:
    """
    Args:
        dtype: Torch data type

    Returns:
        Number of bits used to represent one number of this type.

    """
    return torch.tensor([], dtype=dtype).element_size() * 8


def bitsize_of(x: torch.Tensor) -> int:
    return x.numel() * x.element_size() * 8
