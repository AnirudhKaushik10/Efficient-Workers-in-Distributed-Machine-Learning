import torch
from Compresssor.TopK import TopKCompressor
 

class QSGDCompressor():

    def __init__(self ):
        super().__init__()
        self.quantum_num = 128
        self.topK = TopKCompressor(0.5)

    def compress(self, tensor):

        #res = self.topK.compress(tensor)
        #tensor = self.topK.decompress(res[0],res[1])
        #print(tensor)
        norm = tensor.norm()
        norm = norm.flatten()
        abs_gradient = tensor.abs()

        level_float = self.quantum_num / norm * abs_gradient
        previous_level = level_float.floor()
        prob = torch.empty_like(tensor).uniform_()
        is_next_level = (prob < (level_float - previous_level))
        new_level = (previous_level + is_next_level)

        sign = tensor.sign()
        tensor_compressed = (new_level * sign)
        tensor_compressed = tensor_compressed
        tensor_compressed = tensor_compressed, norm

        return tensor_compressed

    def decompress(self, tensor_compressed):
        tensor_compressed, norm = tensor_compressed

        decode_output = tensor_compressed.type(torch.float32)
        tensor_decompressed = norm / self.quantum_num * decode_output
        #tensor_decompressed = tensor_decompressed.view(shape)
        return tensor_decompressed
