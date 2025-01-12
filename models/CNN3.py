from torch import nn, Size
from models.custom_cnn import CustomCNN
from models.BasicBlock import BasicBlock

class CNN3(CustomCNN):
    def __init__(self, args: dict[str, any], class_count: int, img_size: Size):
        super(CNN3, self).__init__(args, class_count, img_size)
        self.last_layer_output = 1024
        self.seq = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BasicBlock(64, 128),
            nn.AvgPool2d(kernel_size=2, stride=2),

            # Basic block with 256 filters and kernel size of 3
            # Max pooling layer with kernel size of 2 and stride of 2
            BasicBlock(128, 256, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # Basic block with 256 filters and kernel size of 3
            # Max pooling layer with kernel size of 2 and stride of 2
            BasicBlock(256, 512, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Basic block with 512 filters and kernel size of 3
            # Max pooling layer with kernel size of 2 and stride of 2
            # BasicBlock(256, 512, stride=2),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            # Basic block with 1024 filters and kernel size of 3
            # Max pooling layer with kernel size of 2 and stride of 2
            BasicBlock(512, 1024),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # Final flattening layers            
            nn.Flatten(),
            nn.LazyLinear(self.last_layer_output),
            nn.ReLU(),
            nn.LazyLinear(class_count),
        )

    def forward(self, x):
        return self.seq(x)