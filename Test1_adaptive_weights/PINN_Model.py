import torch.nn as nn
import torch
import numpy as np

class PINN_Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers):
        super(PINN_Model, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_layers[0]))
        layers.append(nn.Tanh())
        
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_layers[-1], output_dim))
        self.model = nn.Sequential(*layers)
        self.apply(self._init_weights)          # initialize all layers

        self.register_buffer('mean', torch.zeros(input_dim))   # placeholders, set below
        self.register_buffer('std',  torch.ones(input_dim))
        
    def set_input_stats(self, xy):                 # xy: [N, 2] training coordinates (x, y)
        self.mean.copy_(xy.mean(dim=0))
        self.std.copy_(xy.std(dim=0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = (x - self.mean) / self.std       # z-score inside the model
        return self.model(x)