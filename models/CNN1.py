from torch import nn, Size
from models.custom_cnn import CustomCNN

class CNN1(CustomCNN):
    def __init__(self, args, class_count: int, img_size: Size):
        super(CNN1, self).__init__(args, class_count, img_size)
        self.last_layer_output = 32
        self.seq = nn.Sequential(
            # Conv layer with 8 filters, kernel size 3 and ReLU activation
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1),
            nn.ReLU(),

            # Max pooling layer with stride 2
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv layer with 16 filters, kernel size 3 and ReLU activation
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1),
            nn.ReLU(),
            # Max pooling layer with stride 2
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.LazyLinear(self.last_layer_output),
            nn.ReLU(),
            nn.LazyLinear(class_count)
        )

    def forward(self, x):
        return self.seq(x)