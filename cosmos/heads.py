import torch
import torch.nn as nn


class ResidualMLP(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=128
    ):
        super().__init__()

        self.fc1 = nn.Linear(
            input_dim,
            hidden_dim
        )

        self.fc2 = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.fc3 = nn.Linear(
            hidden_dim,
            output_dim
        )

        self.activation = nn.ReLU()

        # Residual projection when input/output differ
        self.skip = nn.Linear(
            input_dim,
            output_dim
        )

    def forward(self, x):

        residual = self.skip(x)

        x = self.activation(
            self.fc1(x)
        )

        x = self.activation(
            self.fc2(x)
        )

        x = self.fc3(x)

        return x + residual