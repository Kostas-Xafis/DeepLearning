from torch import nn
from models.custom_cnn import CustomCNN
from models.BasicBlock import BasicBlock

class CNN3(CustomCNN):
    def __init__(self, args: dict[str, any], class_count: int):
        super(CNN3, self).__init__(args, class_count)
        self.last_layer_output = 1024
        self.seq = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            BasicBlock(64, 256),
            BasicBlock(256, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            BasicBlock(64, 256, stride=2),
            BasicBlock(256, 64),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 512),
            nn.MaxPool2d(kernel_size=3, stride=2),

            BasicBlock(512, 1024),
            nn.MaxPool2d(kernel_size=2, stride=2),
                        
            nn.Flatten(),
            nn.LazyLinear(self.last_layer_output),
            nn.ReLU(),
            nn.LazyLinear(class_count),
        )

    def forward(self, x):
        return self.seq(x)