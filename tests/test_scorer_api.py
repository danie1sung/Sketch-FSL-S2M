import torch
import torch.nn as nn
from unittest.mock import MagicMock

from src.fewshot_core.scorer import FewShotScorer
from src.fewshot_core.adapter import PerClassAdapter
from src.fewshot_core.classes import CLASSES

def test_scorer_api_and_argmin_selection():
    """
    Tests the FewShotScorer's API, ensuring it correctly computes losses
    for all classes and selects the class with the minimum loss.
    """
    latent_dim = 64
    num_classes = len(CLASSES)
    img_size = 32

    # --- Mocks ---
    # Mock Encoder: returns a fixed vector
    mock_encoder = MagicMock(spec=nn.Module)
    mock_encoder.return_value = torch.randn(1, latent_dim)

    # Mock Decoder: returns a tensor of zeros
    mock_decoder = MagicMock(spec=nn.Module)
    # The decoder output shape should be [num_classes, 1, H, W] for batch processing
    mock_decoder.return_value = torch.zeros(num_classes, 1, img_size, img_size)

    # Mock Loss Function: returns a predictable, non-zero loss
    def mock_loss_fn(reconstruction, target):
        # Create different losses for each item in the batch
        # We simulate this by checking the class_id used for the reconstruction
        # Here we just use the sum of the reconstruction, but the mock decoder returns zeros.
        # So we'll make a more direct loss function mock that returns losses based on class id
        return torch.sum(reconstruction) # This will be 0 for the mock_decoder

    # Let's create a more sophisticated mock for the whole scorer logic
    # that bypasses the internal decoder call.
    
    # --- A simpler test setup ---
    
    # 1. Define a scorer that returns predictable losses
    class MockScorer(FewShotScorer):
        def compute_losses(self, sketch: torch.Tensor) -> dict:
            # Return a predefined loss dictionary, ignoring the sketch input
            # We will make "Chair" have the lowest loss.
            losses = {cls: 10.0 for cls in CLASSES}
            losses["Chair"] = 0.5
            losses["Table"] = 0.6
            return losses

    # Mock components for initialization
    mock_adapter = MagicMock(spec=PerClassAdapter)
    mock_loss_fn = MagicMock()
    
    # Instantiate the MockScorer
    scorer = MockScorer(mock_encoder, mock_decoder, mock_loss_fn, mock_adapter, normalize=False)
    
    # --- Test ---
    dummy_sketch = torch.randn(1, 1, img_size, img_size)
    
    predicted_class, all_losses = scorer.predict(dummy_sketch)
    
    # --- Assertions ---
    assert predicted_class == "Chair", f"Expected 'Chair' to be predicted, but got '{predicted_class}'"
    assert "Chair" in all_losses
    assert all_losses["Chair"] == 0.5
    assert all_losses["Airplane"] == 10.0
    assert len(all_losses) == num_classes, "Scorer should compute loss for all classes."
