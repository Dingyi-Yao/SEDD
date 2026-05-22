import numpy as np
import torch
import torch.nn as nn


class StyleExtractor_conv1(nn.Module):
    def __init__(self, inputsize):
        super(StyleExtractor_conv1, self).__init__()
        
        self.hidden = nn.Linear(inputsize, inputsize//2)
        self.hidden2 = nn.Linear(inputsize//2, inputsize//4)
        self.output = nn.Linear(inputsize//4, 64)
        self.leakyrelu = nn.LeakyReLU()
        
    def forward(self, x):
        x = self.hidden(x)
        x = self.leakyrelu(x)
        x = self.hidden2(x)
        x = self.leakyrelu(x)
        x = self.output(x)

        return x
