import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out += residual
        out = self.relu(out)
        return out

class ConnectFourCNN(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResidualBlock(32),
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(32, 1, 1),
            nn.Flatten(),
            nn.Linear(6*7, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(32, 1, 1),
            nn.Flatten(),
            nn.Linear(6*7, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv_block(x)
        return self.policy_head(x), self.value_head(x)
    
    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

if __name__ == "__main__":
    batch_size = 4
    input_channels = 3
    model = ConnectFourCNN(input_channels=input_channels)
    sample_input = torch.randn(batch_size, input_channels, 6, 7)
    policy_logits, value = model(sample_input)
    print("Policy shape:", policy_logits.shape)  # Should be [4, 7]
    print("Value shape:", value.shape)           # Should be [4, 1]