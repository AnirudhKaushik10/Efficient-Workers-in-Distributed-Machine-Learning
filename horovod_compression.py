import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data.distributed
import horovod.torch as hvd
import horovod.torch.compression as compression

class QSGDCompressor(compression.Compressor):

    def __init__(self ):
        super().__init__()
        self.quantum_num = 128
    @staticmethod
    def compress( tensor):
        #tensor = tensor.flatten()

        norm = tensor.norm()
        norm = norm.flatten()
        abs_gradient = tensor.abs()

        level_float = 128 / norm * abs_gradient
        previous_level = level_float.floor()
        prob = torch.empty_like(tensor).uniform_()
        is_next_level = (prob < (level_float - previous_level))
        new_level = (previous_level + is_next_level)

        sign = tensor.sign()
        tensor_compressed = (new_level * sign)
        tensor_compressed = tensor_compressed
        tensor_compressed = tensor_compressed, norm
        print(norm)
        return tensor_compressed[0], tensor_compressed[1]
    @staticmethod
    def decompress(tensor_compressed, ctx):
        print(ctx)
        tensor_compressed, norm = tensor_compressed, ctx
        decode_output = tensor_compressed.type(torch.float32)
        tensor_decompressed = norm / 128 * decode_output
        #tensor_decompressed = tensor_decompressed.view(shape)
        return tensor_decompressed
