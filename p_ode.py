from torch import nn
import torch


class ProbabilityFlowODE(nn.Module):
    def __init__(self, model, B, shape, epsilon_dist="normal"):
        super().__init__()
        self.model = model
        self.B = B
        self.shape = shape
        self.epsilon_dist = epsilon_dist

    def forward(self, t, state):
        z, _ = state
        
        # Generate random vector every time
        if self.epsilon_dist == "normal":
            epsilon = torch.randn(self.B, *self.shape, device=z.device)
        elif self.epsilon_dist == "rademacher":
            epsilon = torch.randint(0, 2, (self.B, *self.shape), device=z.device) * 2 - 1
            epsilon = epsilon.float()
        else:
            raise ValueError(f"Unknown epsilon_dist: {self.epsilon_dist}")

        with torch.enable_grad():
            t_batch = torch.full((self.B,), t.item(), device=z.device, requires_grad=True)
            z.requires_grad_(True)

            gamma_t, _ = torch.autograd.functional.jvp(
                self.model.gamma,
                (t_batch,),
                (torch.ones_like(t_batch),)
            )

            sigma_t = gamma_t.sigmoid().sqrt()

            pred_noise = self.model(z, t_batch)

            while len(sigma_t.shape) < len(z.shape):
                sigma_t = sigma_t.unsqueeze(-1)

            score = -pred_noise / (sigma_t + 1e-8)
            dz_dt = 0.5 * (-sigma_t * z + score) * sigma_t

            # Hutchinson Trace Estimator: d(log p)/dt = -Tr(df/dz)
            vjp = torch.autograd.grad(dz_dt, z, epsilon, create_graph=False)[0]
            dlogp_dt = -torch.sum(vjp.flatten(1) * epsilon.flatten(1), dim=1, keepdim=True)

        return dz_dt.detach(), dlogp_dt.detach()