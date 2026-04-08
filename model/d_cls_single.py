import torch.nn as nn
import math
import torch
from einops import reduce, rearrange
from einops.layers.torch import Reduce, Rearrange

class Contiguous(torch.nn.Module):
    def __init__(self):
        super(Contiguous, self).__init__()

    def forward(self,x):
        return x.contiguous()

class class_discriminator(nn.Module):
    def __init__(self, num_class=60, drop_out=False):
        super(class_discriminator, self).__init__()
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))

        self.pred_fusion = torch.nn.Sequential(
            Rearrange('b s c -> b c s'),
            Contiguous(),
            torch.nn.AdaptiveAvgPool1d((1)),
            Rearrange('b c 1 -> b c'))

    def forward(self, x):
        if len(x.size()) == 5:
            N, M, C, T, V = x.size()
            x = x.view(N, M, C, -1)
            x = x.mean(3).mean(1)
            x = self.drop_out(x)
            x = self.fc(x)
        elif len(x.size()) == 4:
            N, M, S, V = x.size()
            x = x.mean(1)
            x = self.drop_out(x)
            x = self.fc(x)
            x = self.pred_fusion(x)
        return x





