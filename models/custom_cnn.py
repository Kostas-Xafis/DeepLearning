import torch
import numpy as np
import math
from torch import nn

class CustomCNN(nn.Module):
    def __init__(self, args, class_count: int, img_size: torch.Size):
        super(CustomCNN, self).__init__()
        self.class_count = class_count
        self.image_size = img_size
        self.batch_size = args.get('batch_size', 64)
        self.epochs = args.get('epochs', 20)
        self.lr = args.get('lr', 10e-3)

    def append_flatten_linear_layers(self):
        output_size = np.array(self.image_size, dtype=np.int64)
        last_output_channels = 0
        for l in self.seq:
            if isinstance(l, nn.Conv2d):
                kernel_size = l.kernel_size[0] 
                stride = l.stride[0]
                padding = l.padding[0]
                output_size = np.floor((output_size - kernel_size + 2 * padding) / stride) + 1
                last_output_channels = l.out_channels
                print('Output size:', output_size)
                print('Last output channels:', last_output_channels)
            elif isinstance(l, nn.MaxPool2d):
                kernel_size = l.kernel_size
                stride = l.stride
                output_size = np.floor((output_size - kernel_size + 2 * 0) / stride) + 1
                print('Output size:', output_size)
        
        
                
        self.seq.add_module('flatten', nn.Flatten())
        self.seq.add_module('fc1', nn.Linear((np.prod(output_size) * last_output_channels).astype(np.int64), self.last_layer_output))
        self.seq.add_module('relu', nn.ReLU())
        self.seq.add_module('fc2', nn.Linear(self.last_layer_output, self.class_count))