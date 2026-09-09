from torch import nn
import torch


class DepthController(nn.Module):
    def __init__(self, hidden_size: int, adapter_type: str) -> None:
        super().__init__()
        self.adapter_type = adapter_type
        
        self.input_layer = nn.Linear(hidden_size, 64)
        self.input_ln = nn.LayerNorm(64)
        self.input_dropout = nn.Dropout(0.05)
        self.act = nn.GELU()
        
        self.hl_1 = nn.Linear(64, 32)
        self.hl_1_ln = nn.LayerNorm(32)
        self.hl_1_dropout = nn.Dropout(0.05)
        
        self.output_layer = nn.Linear(32, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        h = self.input_ln(h)
        h = self.input_dropout(h)
        h = self.act(h)
        
        h = self.hl_1(h)
        h = self.hl_1_ln(h)
        h = self.hl_1_dropout(h)
        h = self.act(h)
        
        out = self.output_layer(h)
        return out

