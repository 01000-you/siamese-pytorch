import torch.nn as nn
from torchvision import models
import torch

class Siamese(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(Siamese, self).__init__()
        self.conv = models.resnet18(pretrained=True)
        num_ftrs = self.conv.fc.in_features
        # num_out = num_ftrs
        num_out = 4096

        self.conv.fc = nn.Linear(num_ftrs, num_out)
        self.linear = nn.Sequential(
                                   nn.Dropout(dropout_rate),
                                   nn.Linear(num_out, num_out),
                                   nn.Dropout(dropout_rate),
                                   nn.Sigmoid())
        self.out = nn.Sequential(
                                nn.Linear(num_out, 1),
                                )


    def forward_one(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        out = torch.abs(out)
        return out

# for test
if __name__ == '__main__':
    net = Siamese()
    print(net)
    print(list(net.parameters()))
