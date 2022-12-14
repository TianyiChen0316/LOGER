import torch
from .lstm import MultiInputLSTM

from core import database

class Step2(torch.nn.Module):
    def __init__(self, out_dim=1, use_edge_input=False, squeeze=False):
        super().__init__()
        self.__out_dim = out_dim
        self.__squeeze = squeeze
        self.__feature_size = database.config.feature_size

        self.__lstm_input_feature_size = self.__feature_size * 2 if use_edge_input else 0

        self.lstm = MultiInputLSTM(
            hidden_size=self.__feature_size,
            in_feature_size=self.__lstm_input_feature_size,
            input_branches=2,
            output_branches=out_dim,
        )

    def forward(self, left, right, input=None):
        h_l, c_l = left.chunk(2, dim=-1)
        h_r, c_r = right.chunk(2, dim=-1)
        h, c = self.lstm(((h_l, c_l), (h_r, c_r)), input)
        res = torch.cat([h, c], dim=-1)
        res = res.view(*res.shape[:-1], self.__feature_size * 2, self.__out_dim)
        if self.__squeeze:
            res = res.squeeze(-1)
        return res
