from torch import nn, Size
from models.custom_cnn import CustomCNN

class CNN2(CustomCNN):
    def __init__(self, args: dict[str, any], class_count: int, img_size: Size):
        super(CNN2, self).__init__(args, class_count, img_size)
        self.last_layer_output = 1024
        self.seq = nn.Sequential(
            # 2 Conv layers with 32 filters and kernel size of 3 with ReLU activation
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.ReLU(),

            # Max pooling layer with kernel size of 2 and stride of 4
            nn.MaxPool2d(kernel_size=4, stride=4),

            # 2 Conv layers with 64 filters and kernel size of 3 with ReLU activation
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(),

            # Max pooling layer with kernel size of 2 and stride of 2
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 2 Conv layers with 128 filters and kernel size of 3 with ReLU activation
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            nn.ReLU(),

            # Max pooling layer with kernel size of 2 and stride of 2
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 3 Conv layers with 256 filters and kernel size of 3 with ReLU activation
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU(),

            # Max pooling layer with kernel size of 2 and stride of 2
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv layer with 512 filters and kernel size of 3 with ReLU activation
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            nn.ReLU(),

            # Max pooling layer with kernel size of 2 and stride of 2
            nn.MaxPool2d(kernel_size=2, stride=2), 

            nn.Flatten(),
            nn.LazyLinear(self.last_layer_output),
            nn.ReLU(),
            nn.LazyLinear(class_count),
        )

    def forward(self, x):
        return self.seq(x)