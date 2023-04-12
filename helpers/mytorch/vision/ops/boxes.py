# from signal import raise_signal
import torch
from . import _box_convert as convert_fns

def box_convert(inp, in_mode, out_mode):
    '''
    Inputs:
        inp: Tensor[...,4] or list[Tensor[...,4]]
        in_mode, out_mode: str. The current implementation supports 'xyxy' and 'xywh' as modes
    Output:
        Returns the converted bboxes. The output type is same as that of the input
    '''
    convert_fn = getattr(convert_fns, '_'+in_mode+'_to_'+out_mode, None)
    if convert_fn is None: 
        raise AttributeError('convert_function is not availbale')

    if torch.is_tensor(inp):
        return convert_fn(inp)
    elif type(inp) == list and torch.is_tensor(inp[0]):
        return [convert_fn(x) for x in inp]
    else:
        raise ValueError("Input has to be a tensor or list of tensors")

def _normalize_tensor(inp, width,height):
        inp[...,0] /= width
        inp[...,1] /= height
        inp[..., 2] /= width
        inp[...,3] /= height
        return inp
    
def normalize_bbox(inp, width, height, width_first = True):
    '''
    inp: Tensor[...,4] or list[Tensor[...,4]].
    height: The height of the image
    width: The width of the image.
    '''
    if not width_first:
        width, height = height, width

    if torch.is_tensor(inp):
        _normalize_tensor(inp, width, height)
    elif type(inp) == list and torch.is_tensor(inp[0]):
        return [_normalize_tensor(b,width,height) for b in inp]
    else: 
        raise ValueError("Input has to be a tensor or list of tensors")