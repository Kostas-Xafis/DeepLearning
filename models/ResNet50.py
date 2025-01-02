from torchvision import models
from models.custom_cnn import CustomCNN

class Resnet50Pretrained(CustomCNN):
    def __init__(self, args: dict[str, any], class_count: int):
        super(Resnet50Pretrained, self).__init__(args, class_count, None)
        self.net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    def to(self, device):
        self.net.to(device)
        return self
    
    def forward(self, x):
        return self.net.forward(x)
    
class Resnet50(CustomCNN):
    def __init__(self, args: dict[str, any], class_count: int):
        super(Resnet50, self).__init__(args, class_count, None)
        self.net = models.resnet50(weights=False)
    
    def to(self, device):
        self.net.to(device)
        return self
    
    def forward(self, x):
        return self.net.forward(x)