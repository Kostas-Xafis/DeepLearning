from torch import nn

class BasicBlock(nn.Module):
    def __init__(self, n_in: int, n_filters: int, stride: int = 1):
        super(BasicBlock, self).__init__()
        if stride != 1 and stride != 2:
            raise ValueError('Stride must be 1 or 2')
        self.n_in = n_in
        self.n_filters = n_filters
        self.stride = stride
        self.relu = nn.ReLU()
        self.seq = nn.Sequential(
            # Conv layer with n_filters filters, kernel size of 3, stride of 1, padding of 1, batch normalization and ReLU activation
            nn.Conv2d(n_in, n_filters, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            
            # Conv layer with n_filters filter and kernel size of 3, stride of 1, batch normalization and ReLU activation
            nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
        )
        
        self.downsample = nn.Sequential(
            nn.Conv2d(n_in, n_filters, kernel_size=3, stride=stride),
            nn.BatchNorm2d(n_filters),
        )

        if stride == 2:
            self.seq2 = nn.Conv2d(n_in, n_filters, kernel_size=1, stride=stride),
            
    
    def forward(self, x):
        identity = x

        if self.n_in != self.n_filters:
            identity = self.downsample(identity)
        
        if self.stride == 2:
            x = self.seq2(x)
        return self.relu(self.seq(x) + identity)
        
        
        