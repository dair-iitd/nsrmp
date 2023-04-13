import torch
import torch.nn as nn


class ActionSimulator(nn.Module):
    def __init__(self, nr_actions, mode = 2):
        """ 
        Mode 1 : Use Only Base Object PosInfo
        Mode 2 : Use Both Base and Move Object PosInfo
        """
        super().__init__()
        self.mode = mode
        self.input_dim = (5 + nr_actions) if self.mode == 1 else (10 + nr_actions)
        self.middle_dim = 256
        self. output_dim = 5
        self.map = nn.Sequential(
            nn.Linear(self.input_dim, self.middle_dim), 
            nn.ReLU(),
            nn.Linear(self.middle_dim,2*self.middle_dim),
            nn.ReLU(),
            nn.Linear(2*self.middle_dim,2*self.middle_dim),
            nn.ReLU(),
            nn.Linear(2*self.middle_dim, self.middle_dim),
            nn.ReLU(),
            nn.Linear(self.middle_dim, self.output_dim),
            nn.Tanh()
        )


    def forward(self, obj1, obj2, actions):
        if self.mode == 2:
            x = torch.cat((obj1, obj2, actions), dim=-1)
        else:
            x = torch.cat((obj2, actions), dim=-1)
        z = self.map(x)

        return z 

    def reset_parameters(self):
        for layer in self.map.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
