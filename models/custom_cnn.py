from torch import nn

class CustomCNN(nn.Module):
    def __init__(self, args, class_count: int):
        super(CustomCNN, self).__init__()
        self.class_count = class_count
        self.batch_size = args.get('batch_size', 64)
        self.epochs = args.get('epochs', 20)
        self.lr = args.get('lr', 10e-3)