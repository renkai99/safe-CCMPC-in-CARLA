import torch

def tensor_to_numpy(x: torch.Tensor):
    return x.cpu().detach().numpy()

def expand_and_repeat(A: torch.Tensor, dim, n):
    """Expand/unsqueeze the tensor and
    repeat it on the expanded dimensions.

    Parameters
    ==========
    A: torch.Tensor
        Tensor to expand and repeat.
    dim: int or list
        Indices of A to add new dimensions to.
    n: int or list
        Sizes to repeat tensor on the expanded dimensions. 

    Returns
    =======
    torch.Tensor
    """
    d = A.dim()
    if isinstance(dim, (list, tuple)):
        tile = [1] * (d + len(dim))
        for i in range(len(dim)):
            tile[dim[i] + i] = n[i]
            A = A.unsqueeze(dim[i] + i)
        return A.repeat(tile)
    else:
        tile = [1] * (d + 1)
        tile[dim] = n
        return A.unsqueeze(dim).repeat(tile)
