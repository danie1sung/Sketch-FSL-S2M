import os
from glob import glob
from typing import Tuple, List, Union
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from ..fewshot_core.classes import CLASS_TO_ID

class SketchDataset(Dataset):
    """
    Dataset for loading sketches. Assumes a directory structure like:
    <root_dir>/
        Airplane/
            sketch-1.png
            ...
        Bench/
            ...
    """
    def __init__(self, root_dir: str, img_size: int = 224):
        self.root_dir = root_dir
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            lambda x: 1 - x,  # Invert colors (black sketch on white background)
            # Simple binarization
            lambda x: (x > 0.1).float(),
        ])
        
        self.file_paths = []
        self.labels = []
        self.class_to_indices = {i: [] for i in range(len(CLASS_TO_ID))}

        if not os.path.isdir(root_dir):
            print(f"WARNING: Dataset directory not found at {root_dir}. This may cause errors.")
            return

        for class_name, class_id in CLASS_TO_ID.items():
            class_dir = os.path.join(self.root_dir, class_name)
            sketches = glob(os.path.join(class_dir, '*.png'))
            
            for i, sketch_path in enumerate(sketches):
                idx = len(self.file_paths)
                self.file_paths.append(sketch_path)
                self.labels.append(class_id)
                self.class_to_indices[class_id].append(idx)

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: Union[int, List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(idx, list):
            # Handle batch indexing for episodic sampler
            sketches = []
            labels = []
            for i in idx:
                sketch, label = self._get_single_item(i)
                sketches.append(sketch)
                labels.append(label)
            return torch.stack(sketches), torch.tensor(labels, dtype=torch.long)
        else:
            return self._get_single_item(idx)

    def _get_single_item(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            sketch_img = Image.open(path).convert('L')
            sketch_tensor = self.transform(sketch_img)
        except Exception as e:
            print(f"Warning: Could not load or process image at {path}. Returning zero tensor. Error: {e}")
            sketch_tensor = torch.zeros((1, self.img_size, self.img_size))

        return sketch_tensor, label

    def get_indices_for_class(self, class_id: int) -> List[int]:
        """
        Returns all dataset indices for a given class ID.
        Required by the EpisodicSampler.
        """
        return self.class_to_indices[class_id]
