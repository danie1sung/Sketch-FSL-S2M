import torch
import torch.nn as nn
import torch.optim as optim

class PerClassAdapter(nn.Module):
    """
    A small, per-class adapter that is calibrated on the support set.
    This example uses a simple linear transformation for each class.
    """
    def __init__(self, latent_dim: int, num_classes: int, hidden_dim: int = 0):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Create a list of small MLPs, one for each class
        if hidden_dim > 0:
            self.adapters = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, latent_dim)
                ) for _ in range(num_classes)
            ])
        else: # Linear adapter
            self.adapters = nn.ModuleList([
                nn.Linear(latent_dim, latent_dim) for _ in range(num_classes)
            ])
        
        # Initialize as identity transformations
        for adapter in self.adapters:
            if isinstance(adapter, nn.Linear):
                adapter.weight.data.copy_(torch.eye(latent_dim))
                adapter.bias.data.fill_(0.0)

    def forward(self, z3d: torch.Tensor, class_ids: torch.Tensor) -> torch.Tensor:
        """
        Apply the class-specific adapter to each latent vector.

        Args:
            z3d (torch.Tensor): Latent vectors of shape [B, latent_dim].
            class_ids (torch.Tensor): Class IDs for each vector, shape [B].

        Returns:
            torch.Tensor: Adapted latent vectors, shape [B, latent_dim].
        """
        output = torch.zeros_like(z3d)
        for i in range(self.num_classes):
            mask = (class_ids == i)
            if mask.any():
                output[mask] = self.adapters[i](z3d[mask])
        return output

    def fit(self, encoder, decoder, loss_fn, support_sketches, support_labels, lr=1e-3, steps=10):
        """
        Fit the adapter on the support set.
        """
        optimizer = optim.Adam(self.parameters(), lr=lr)
        # The encoder now returns both shape and view latents
        zs, zv_pred = encoder(support_sketches)
        z3d = zs.detach()
        zv_pred = zv_pred.detach()


        for _ in range(steps):
            optimizer.zero_grad()
            
            adapted_z3d = self.forward(z3d, support_labels)
            # The decoder now requires the view latent to render the reconstruction
            reconstructions = decoder(adapted_z3d, support_labels, zv=zv_pred)
            
            loss = loss_fn(reconstructions, support_sketches)
            
            loss.backward()
            optimizer.step()
