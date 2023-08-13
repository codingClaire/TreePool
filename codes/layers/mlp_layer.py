import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_ft, out_ft, act='prelu', bias=True):
        super().__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=bias)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        x_fts = self.fc(x)
        if self.bias is not None:
            x_fts += self.bias
        return self.act(x_fts)
