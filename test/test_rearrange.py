import torch
from torch.functional import rearrange

def test_rearrange_identity():
    x = torch.randn(2, 3, 4)
    out = rearrange(x, "b c h -> b c h")
    assert torch.equal(out, x)
