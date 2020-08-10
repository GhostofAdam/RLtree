import torch
import torch.nn as nn
import torch.nn.functional as F
class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers):
        super(RNN, self).__init__()
        self.unit_h = nn.Linear(hidden_size,hidden_size)
        self.unit_i = nn.Linear(input_size,hidden_size)
        self.unit_o = nn.Linear(hidden_size,input_size)
        
    def forward(self, h, x):
        h = F.tanh(self.unit_h(h)+self.unit_i(x))
        o = F.tanh(self.unit_o(h))
        return h,o