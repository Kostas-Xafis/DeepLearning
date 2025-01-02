from torch import nn
from models.BasicBlock import BasicBlock
from models.custom_cnn import CustomCNN

class CNN3(CustomCNN):
    def __init__(self, args: dict[str, any], class_count: int, img_size: tuple[int, int, int]):
        super(CNN3, self).__init__(args, class_count, img_size)
        self.last_layer_output = 1024
        self.seq = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            BasicBlock(64, 64),
            BasicBlock(64, 64),
            BasicBlock(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Basic block with 256 filters and kernel size of 3
            # Max pooling layer with kernel size of 2 and stride of 2
            BasicBlock(64, 128),
            BasicBlock(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Basic block with 256 filters and kernel size of 3
            # Max pooling layer with kernel size of 2 and stride of 2
            BasicBlock(128, 256),
            BasicBlock(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Basic block with 512 filters and kernel size of 3
            # Max pooling layer with kernel size of 2 and stride of 2
            BasicBlock(256, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.append_flatten_linear_layers(self.seq)
        # Add final softmax output layer
        self.seq.add_module('sf', nn.Softmax(dim=1))

    def forward(self, x):
        return self.seq(x)