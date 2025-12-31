import torch
import torch.nn.functional as F

def silhouette_l2_loss(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    A simple L2 loss between the predicted silhouette and a target silhouette.
    This serves as a proxy for more complex 3D reconstruction losses.

    Args:
        reconstruction (torch.Tensor): The decoder's output, shape [B, 1, H, W].
        target (torch.Tensor): The ground-truth proxy (e.g., the binarized input sketch), shape [B, 1, H, W].

    Returns:
        torch.Tensor: A scalar loss value.
    """
    print("Shape of reconstruction:", reconstruction.shape)
    print("Shape of target:", target.shape)
    return F.mse_loss(reconstruction, target)

def get_loss_function(loss_type: str):
    """
    Factory function to get the desired loss function.
    """
    if loss_type == "silhouette_l2":
        return silhouette_l2_loss
    elif loss_type == "chamfer":
        # Placeholder for a Chamfer distance implementation
        raise NotImplementedError("Chamfer distance loss is not yet implemented.")
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
