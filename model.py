import torch
import torch.nn as nn


class WNet(nn.Module):
    def __init__(
        self,
        layer_num: int,
        hidden_size: int,
        out_dim: int,
        in_features: int,
        act: str = "gelu",
    ):
        super().__init__()
        act = act.lower().strip()
        if act not in {"tanh", "gelu", "gelu_tanh", "silu"}:
            raise ValueError("act must be one of: tanh, gelu, gelu_tanh, silu")

        self.act_name = act

        def make_act():
            if act == "tanh":
                return nn.Tanh()
            if act == "gelu":
                return nn.GELU()
            if act == "gelu_tanh":
                return nn.GELU(approximate="tanh")
            if act == "silu":
                return nn.SiLU()
            raise RuntimeError("unreachable")

        layers = []
        for i in range(layer_num - 1):
            layers.append(nn.Linear(in_features if i == 0 else hidden_size, hidden_size))
            layers.append(make_act())
        layers.append(nn.Linear(hidden_size, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)