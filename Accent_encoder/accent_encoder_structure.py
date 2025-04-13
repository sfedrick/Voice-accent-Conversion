import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(ConvBlock, self).__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=padding, dilation=dilation)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super(SelfAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        return self.norm(attn_output + x)

class AcousticEncoder(nn.Module):
    def __init__(self, in_features, hidden_dim=256):
        super(AcousticEncoder, self).__init__()
        self.conv1 = ConvBlock(in_features, hidden_dim, kernel_size=5, dilation=1)
        self.conv2 = ConvBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2)
        self.conv3 = ConvBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=1)
        self.conv4 = ConvBlock(hidden_dim, hidden_dim, kernel_size=1, dilation=1)

        self.attn = SelfAttention(embed_dim=hidden_dim, num_heads=4)

    def forward(self, x):
        x = x.transpose(1, 2)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.transpose(1, 2)
        x = self.attn(x)

       
        x = x.mean(dim=1)  
        return x  #

# Example usage
if __name__ == "__main__":
    batch_size = 8
    time_steps = 100  
    input_features = 20  # MFCC (13) + periodicity (7)

    model = AcousticEncoder(in_features=input_features)
    dummy_input = torch.randn(batch_size, time_steps, input_features)
    output = model(dummy_input)
    print(output.shape)  