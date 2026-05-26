import torch
from torch import nn
from torch.nn import functional as F

class PositiveLinear(nn.Module):
    """
    A linear layer where weights are constrained to be strictly positive.
    This guarantees that if the input increases, the output also increases.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, x):
        # Enforce positive weights using softplus
        positive_weight = F.softplus(self.weight)
        return F.linear(x, positive_weight, self.bias)

class MonotonicScheduleNet(nn.Module):
    """
    Learns a monotonically decreasing log-SNR schedule gamma(t).
    Based on the Variational Diffusion Models (Kingma et al., 2021).
    """
    def __init__(self, hidden_dim=32):
        super().__init__()
        # We don't use Fourier features for the schedule network itself, 
        # because Fourier features oscillate (destroying monotonicity).
        # We pass the scalar t directly into the monotonic network.
        
        self.l1 = PositiveLinear(1, hidden_dim)
        self.l2 = PositiveLinear(hidden_dim, hidden_dim)
        self.l3 = PositiveLinear(hidden_dim, 1)
        
    def forward(self, t):
        """
        Args:
            t: Tensor of shape (batch_size, 1) with values in [0, 1]
        Returns:
            gamma(t): The log-SNR at time t. 
        """
        # Ensure t is shape (batch_size, 1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
            
        x = self.l1(t)
        x = torch.sigmoid(x) # Sigmoid is monotonically increasing
        
        x = self.l2(x)
        x = torch.sigmoid(x)
        
        x = self.l3(x)
        
        # The network outputs a monotonically *increasing* value with t.
        # We negate it so that the log-SNR strictly *decreases* as t goes from 0 to 1.
        # In VDMs, log-SNR is highest at t=0 (pure data) and lowest at t=1 (pure noise).
        gamma_t = -x 
        
        return gamma_t