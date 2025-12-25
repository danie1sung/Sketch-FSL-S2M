import torch
import pytest
from unittest.mock import MagicMock

from src.pipeline.train_fewshot import run_episode
from src.sketch2model_io.encoder import Sketch2ModelEncoder
from src.sketch2model_io.decoder import Sketch2ModelDecoder
from src.fewshot_core.adapter import PerClassAdapter
from src.fewshot_core.scorer import FewShotScorer
from src.fewshot_core.losses import silhouette_l2_loss

@pytest.mark.skip(reason="This is a more complex integration test that requires a synthetic dataset. Skipping for now.")
def test_episode_loop_runs_end_to_end():
    """
    A smoke test to ensure a single training episode runs without crashing on CPU.
    It uses real components with random weights.
    """
    # --- Parameters ---
    n_way = 3
    k_shot = 2
    q_query = 3
    latent_dim = 32
    img_size = 32
    num_classes = 13
    device = torch.device("cpu")

    # --- Models (Real components, random weights) ---
    encoder = Sketch2ModelEncoder(latent_dim=latent_dim).to(device)
    decoder = Sketch2ModelDecoder(latent_dim=latent_dim, num_classes=num_classes, output_size=img_size).to(device)
    adapter = PerClassAdapter(latent_dim=latent_dim, num_classes=num_classes).to(device)
    
    scorer = FewShotScorer(
        encoder,
        decoder,
        silhouette_l2_loss,
        adapter,
        normalize=True
    )
    
    # --- Synthetic Data ---
    # Create a synthetic episode dictionary
    support_classes = [0, 1, 2] # Airplane, Bench, Cabinet
    
    # Support Set: N*K items
    support_sketches = torch.rand(n_way * k_shot, 1, img_size, img_size)
    support_labels = torch.tensor([c for c in support_classes for _ in range(k_shot)])
    
    # Query Set: N*Q items
    query_sketches = torch.rand(n_way * q_query, 1, img_size, img_size)
    query_labels = torch.tensor([c for c in support_classes for _ in range(q_query)])

    episode = {
        'support_set': (support_sketches, support_labels),
        'query_set': (query_sketches, query_labels),
        'support_classes': support_classes
    }

    # --- Run Episode ---
    try:
        accuracy, correct, total = run_episode(episode, scorer, device)
        
        # --- Assertions ---
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0
        assert correct <= total
        assert total == n_way * q_query

    except Exception as e:
        pytest.fail(f"The episode loop failed with an exception: {e}")
