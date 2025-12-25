import torch

def basic_collate_fn(batch):
    """
    A basic collate function that stacks tensors for sketches and labels.
    Assumes each item in the batch is a tuple (sketch_tensor, label_id).
    """
    sketches = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return sketches, labels
