from typing import Dict, Tuple, List
import torch
import numpy as np

from .classes import CLASSES, CLASS_TO_ID
from .adapter import PerClassAdapter

class FewShotScorer:
    """
    Computes per-class reconstruction losses and predicts the class with the minimum loss.
    Optionally calibrates loss statistics on a support set for normalization.
    """
    def __init__(self, encoder, decoder, loss_fn, adapter: PerClassAdapter, normalize: bool = False, classes: list = None):
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.adapter = adapter
        self.normalize = normalize

        # Determine which classes to consider for scoring/prediction
        from .classes import CLASSES as ALL_CLASSES, CLASS_TO_ID
        self.classes = classes if classes is not None else ALL_CLASSES
        self.num_classes = len(self.classes)

        # Map global class IDs to local adapter indices (0..num_classes-1)
        self._global_to_local = {CLASS_TO_ID[name]: i for i, name in enumerate(self.classes)}

        # Statistics for normalization, calibrated on the support set.
        self.loss_mean = torch.zeros(self.num_classes)
        self.loss_std = torch.ones(self.num_classes)

    def calibrate(self, support_sketches: torch.Tensor, support_labels: torch.Tensor):
        """
        Calibrate normalization statistics (mean, std) on the support set losses.
        Also fits the per-class adapter on the support set.
        """
        # 1. Map support labels (global IDs) to local indices and fit the adapter
        support_labels_local = torch.tensor(
            [self._global_to_local[int(l.item())] for l in support_labels],
            dtype=torch.long,
            device=support_sketches.device
        )
        self.adapter.fit(self.encoder, self.decoder, self.loss_fn, support_sketches, support_labels_local)

        # 2. Calibrate normalization statistics
        if self.normalize:
            all_losses = {i: [] for i in range(self.num_classes)}
            
            with torch.no_grad():
                z3d = self.encoder(support_sketches)
                for i in range(len(support_sketches)):
                    global_id = int(support_labels[i].item())
                    local_id = self._global_to_local[global_id]
                    
                    # Get adapted latent code for this class
                    adapted_z3d = self.adapter(z3d[i].unsqueeze(0), torch.tensor([local_id], device=z3d.device))

                    loss = self.loss_fn(
                        self.decoder(adapted_z3d, torch.tensor([local_id], device=z3d.device)),
                        support_sketches[i].unsqueeze(0)
                    )
                    all_losses[local_id].append(loss.item())

            for i in range(self.num_classes):
                if all_losses[i]:
                    self.loss_mean[i] = np.mean(all_losses[i])
                    self.loss_std[i] = np.std(all_losses[i]) if len(all_losses[i]) > 1 else 1.0
    
    @torch.no_grad()
    def compute_losses(self, sketch: torch.Tensor) -> Dict[str, float]:
        """
        For a single sketch, compute the reconstruction loss for every possible class.

        Args:
            sketch (torch.Tensor): A single sketch tensor of shape [1, 1, H, W].

        Returns:
            Dict[str, float]: A dictionary mapping class names to their reconstruction losses.
        """
        z3d = self.encoder(sketch)
        losses = {}

        class_ids = torch.arange(self.num_classes, device=z3d.device)
        
        # Get adapted latent codes for all classes
        adapted_z3d = self.adapter(z3d.repeat(self.num_classes, 1), class_ids)

        reconstructions = self.decoder(adapted_z3d, class_ids)
        target = sketch.repeat(self.num_classes, 1, 1, 1)
        
        batch_losses = torch.mean((reconstructions - target).pow(2), dim=[1,2,3])

        for i, class_name in enumerate(self.classes):
            raw_loss = batch_losses[i].item()
            
            # Normalize if required
            if self.normalize:
                mean = self.loss_mean[i].item()
                std = self.loss_std[i].item()
                # Add epsilon to std to prevent division by zero.
                normalized_loss = (raw_loss - mean) / max(std, 1e-6)
                losses[class_name] = normalized_loss
            else:
                losses[class_name] = raw_loss
                
        return losses

    def predict(self, sketch: torch.Tensor) -> Tuple[str, Dict[str, float]]:
        """
        Predict the class of a sketch by finding the minimum reconstruction loss.

        Args:
            sketch (torch.Tensor): A single sketch tensor of shape [1, 1, H, W].

        Returns:
            Tuple[str, Dict[str, float]]: The predicted class name and the dictionary of all class losses.
        """
        losses = self.compute_losses(sketch)
        
        # Find the class with the minimum loss
        pred_class = min(losses, key=losses.get)
        
        return pred_class, losses
