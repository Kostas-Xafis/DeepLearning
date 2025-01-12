from torch import nn, Size

class CustomCNN(nn.Module):
    def __init__(self, args, class_count: int, img_size: Size):
        super(CustomCNN, self).__init__()
        self.class_count = class_count
        self.image_size = img_size
        self.batch_size = args.get('batch_size', 64)
        self.epochs = args.get('epochs', 20)
        self.lr = args.get('lr', 10e-3)