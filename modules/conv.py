import torch
import torch.nn as nn


class Linear(nn.Linear):
    def __init__(self,
                 in_dim,
                 out_dim,
                 bias=True,
                 w_init_gain='linear'):
        super(Linear, self).__init__(in_dim,
                                     out_dim,
                                     bias)
        nn.init.xavier_uniform_(self.weight,
                                gain=nn.init.calculate_gain(w_init_gain))



class Conv1d(nn.Conv1d):
    def __init__(self, *args, activation=None, **kwargs):
        super(Conv1d, self).__init__(*args, **kwargs)
        self.padding = (self.dilation[0]*(self.kernel_size[0]-1))//2
        self.act=None
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('linear'))
        
        if not activation is None:
            self.act = activation
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, inputs, mask=None):
        if self.act is None:
            outputs = super(Conv1d, self).forward(inputs)
        else:
            outputs = self.act(super(Conv1d, self).forward(inputs))
        
        if mask is None:
            return outputs
        else:
            outputs = outputs.masked_fill(mask.unsqueeze(1), 0)
            return outputs
