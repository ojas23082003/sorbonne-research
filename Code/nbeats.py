import torch
import torch.nn as nn

class NBeatsBlock(nn.Module):
    def __init__(self, input_size, forecast_size, hidden_size, n_layers):
        super().__init__()
        layers = []
        actual_input = input_size
        for _ in range(n_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        self.fc_stack = nn.Sequential(*layers)
        self.backcast_linear = nn.Linear(hidden_size, hidden_size)
        self.forecast_linear = nn.Linear(hidden_size, hidden_size)
        self.backcast_basis = nn.Linear(hidden_size, actual_input)
        self.forecast_basis = nn.Linear(hidden_size, forecast_size)

    def forward(self, x):
        x = self.fc_stack(x)
        theta_b = self.backcast_linear(x)
        theta_f = self.forecast_linear(x)
        backcast = self.backcast_basis(theta_b)
        forecast = self.forecast_basis(theta_f)
        return backcast, forecast

class NBeats(nn.Module):
    def __init__(self, input_size, forecast_size,
                 hidden_size=512, n_blocks=3, n_layers=4, theta_size=None):
        super().__init__()
        # if theta_size is None:
        #     theta_size = forecast_size
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_size, forecast_size, hidden_size, n_layers)
            for _ in range(n_blocks)
        ])
        self.input_size = input_size
        self.forecast_size = forecast_size

    def forward(self, x):
        forecast = torch.zeros(x.size(0), self.forecast_size, device=x.device)
        for block in self.blocks:
            backcast, block_forecast = block(
                x
            )
            x = x - backcast  # residual backcast stacking
            forecast = forecast + block_forecast  # aggregate forecast
        return forecast
