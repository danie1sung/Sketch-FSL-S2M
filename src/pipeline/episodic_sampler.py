import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Any, List, Iterator

from ..fewshot_core.classes import CLASSES, CLASS_TO_ID

class EpisodicSampler:
    """
    A deterministic sampler that creates N-way K-shot Q-query episodes
    from a source dataset.
    """
    def __init__(self, dataset: Dataset, n_way: int, k_shot: int, q_query: int, num_episodes: int):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.num_episodes = num_episodes

        if not hasattr(dataset, 'get_indices_for_class'):
            raise AttributeError("The provided dataset must have a 'get_indices_for_class' method.")

        self.class_indices = {
            i: self.dataset.get_indices_for_class(i)
            for i in range(len(CLASSES))
        }

    def __len__(self) -> int:
        return self.num_episodes

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Yields a single few-shot episode.
        """
        # Use a fixed seed for reproducibility of episodes
        rng = np.random.default_rng(seed=42)

        for _ in range(self.num_episodes):
            # 1. Sample N classes for the episode
            episode_classes = rng.choice(len(CLASSES), size=self.n_way, replace=False)
            
            support_indices = []
            query_indices = []
            
            for class_id in episode_classes:
                # 2. Sample K+Q items from each class
                num_samples = self.k_shot + self.q_query
                class_samples = rng.choice(
                    self.class_indices[class_id],
                    size=num_samples,
                    replace=False # No replacement within an episode
                )
                
                support_indices.extend(class_samples[:self.k_shot])
                query_indices.extend(class_samples[self.k_shot:])

            # Collate data for the episode
            support_set = self.dataset[support_indices]
            query_set = self.dataset[query_indices]

            yield {
                "support_classes": episode_classes.tolist(),
                "support_set": support_set,
                "query_set": query_set,
            }
