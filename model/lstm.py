import torch
import torch.nn.functional as F

class MultiInputLSTM(torch.nn.Module):
    def __init__(self, hidden_size, in_feature_size=None, input_branches=2, output_branches=1):
        super().__init__()
        self.out_feature_size = hidden_size
        self.input_branches = input_branches
        self.output_branches = output_branches

        if in_feature_size is None:
            in_feature_size = 0#hidden_size
        self.FC = torch.nn.Linear(hidden_size * input_branches + in_feature_size, (input_branches + 3) * hidden_size * output_branches)
        #print('shape', hidden_size * input_branches + in_feature_size)

        self.LNh = torch.nn.LayerNorm(hidden_size, )
        self.LNc = torch.nn.LayerNorm(hidden_size, )

    def forward(self, branches, input=None):
        # branches = [(h, c), ...]
        hs, cs = zip(*branches)
        if input is not None:
            fc_input = torch.cat([*hs, input], dim=-1)
        else:
            fc_input = torch.cat(hs, dim=-1)
        #print(fc_input.shape)
        lstm_in = self.FC(fc_input)
        a, i, o, *fs = lstm_in.chunk(self.input_branches + 3, -1)
        c = a.tanh() * i.sigmoid()
        for f, _c in zip(fs, cs):
            _c = _c.repeat(*(1 for i in range(_c.ndim - 1)), self.output_branches)
            c = c + f.sigmoid() * _c
        h = o.sigmoid() * c.tanh()
        return h, c

class GatedCell(torch.nn.Module):
    """
    Identical to LSTM cell when cell input is 0.
    """
    def __init__(self, in_feature_size, out_feature_size=None):
        super().__init__()
        if out_feature_size is None:
            out_feature_size = in_feature_size
        self.o_fc = torch.nn.Linear(in_feature_size, out_feature_size, bias=False)
        self.c_fc = torch.nn.Linear(in_feature_size, out_feature_size, bias=False)

    def forward(self, input):
        return F.sigmoid(self.o_fc(input)) * F.tanh(self.c_fc(input))
