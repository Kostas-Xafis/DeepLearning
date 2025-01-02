import numpy as np
from torch import nn, Size

from models import BasicBlock

class CustomCNN(nn.Module):
    def __init__(self, args, class_count: int, img_size: Size):
        super(CustomCNN, self).__init__()
        self.class_count = class_count
        self.image_size = img_size
        self.batch_size = args.get('batch_size', 64)
        self.epochs = args.get('epochs', 20)
        self.lr = args.get('lr', 10e-3)

    def _unfold_sequence(self, seq: nn.Sequential):
        full_seq = []
        for l in seq:
            if isinstance(l, nn.Sequential):
                full_seq.extend(self._unfold_sequence(l))
            elif isinstance(l, BasicBlock.BasicBlock):
                full_seq.extend(self._unfold_sequence(l.seq))
            else:
                full_seq.append(l)
        return full_seq

    def append_flatten_linear_layers(self, seq: nn.Sequential):
        output_size = np.array(self.image_size, dtype=np.int64)
        last_output_channels = 0

        for i, l in enumerate(self._unfold_sequence(seq)):
            # print('Layer:', i, l)
            if isinstance(l, nn.Conv2d):
                kernel_size = l.kernel_size[0] 
                stride = l.stride[0]
                padding = l.padding[0]
                output_size = np.floor((output_size - kernel_size + 2 * padding) / stride) + 1
                last_output_channels = l.out_channels
            elif isinstance(l, nn.MaxPool2d) or isinstance(l, nn.AvgPool2d):
                kernel_size = l.kernel_size
                stride = l.stride
                output_size = np.floor((output_size - kernel_size + 2 * 0) / stride) + 1
        
        flatten_size_input = (np.prod(output_size) * last_output_channels).astype(np.int64)
        seq.add_module('flatten', nn.Flatten())
        seq.add_module('fc1', nn.Linear(flatten_size_input, self.last_layer_output))
        seq.add_module('relu', nn.ReLU())
        seq.add_module('fc2', nn.Linear(self.last_layer_output, self.class_count))