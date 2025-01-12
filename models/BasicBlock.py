import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, n_filters, stride=1):
        super(BasicBlock, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, n_filters, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(n_filters),
        )
        
        # Downsampling layer, if needed
        self.downsample = None
        if in_channels != n_filters:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, n_filters, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(n_filters),
            )
        self.seq2 = None
        if stride != 1:
            self.seq2 = nn.Sequential(
                nn.Conv2d(in_channels, n_filters, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(n_filters),
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        
        out = self.seq(x)
        
        # Downsampling if needed
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        if self.seq2 is not None:
            out2 = self.seq2(x)
            out += out2
        
        out = self.relu(out)
        
        return out

# Example usage:
# block = BasicBlock(in_channels=64, n_filters=64, stride=2)
# print(block)
