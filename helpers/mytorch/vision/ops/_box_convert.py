import torch

__all__ = ['_xywh_to_xyxy', '_xyxy_to_xywh', '_yxhw_to_xywh']

def _xywh_to_xyxy(inp):
    out = torch.clone(inp)
    out[...,0:2] = inp[...,0:2]
    out[...,2:4] = inp[...,0:2] + inp[..., 2:4]
    return out

def _xyxy_to_xywh(inp):
    out = torch.clone(inp)
    out[...,0:2] = inp[...,0:2]
    out[...,2:4] = inp[...,2:4] - inp[...,0:2]
    return out

def _yxhw_to_xywh(inp):
    out = torch.clone(inp)
    out[...,0] = inp[...,1]
    out[...,1] = inp[...,0]
    out[...,2] = inp[...,3]
    out[...,3] = inp[...,2]
    return out

