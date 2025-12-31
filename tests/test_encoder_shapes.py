import torch
import pytest

from src.sketch2model_core.models.view_disentangle_model import ViewDisentangleModel
from src.sketch2model_core.wrappers import EncoderWrapper

class Opt:
    def __init__(self):
        self.name = 'chair_pretrained'
        self.checkpoints_dir = './checkpoints'
        self.summary_dir = './runs'
        self.seed = 0
        self.class_id = 'chair'
        self.model = 'view_disentangle'
        self.dim_in = 3
        self.grl_lambda = 1
        self.n_vertices = 642
        self.image_size = 224
        self.view_dim = 512
        self.template_path = 'sketch2model-fewshot/src/sketch2model_core/SoftRas/soft_renderer/data/obj/sphere/sphere_642.obj'
        self.dataset_mode = 'shapenet'
        self.dataset_root = 'load/shapenet-synthetic'
        self.num_threads = 4
        self.batch_size = 64
        self.max_dataset_size = float("inf")
        self.phase = 'test'
        self.load_epoch = 'latest'
        self.verbose = False
        self.suffix = ''
        self.isTrain = False
        self.isTest = True
        self.isInfer = False
        self.n_gpus = 0
        self.device = 'cpu'
        self.vis_freq = 1
        self.val_epoch_vis_n = 1
        self.test_epoch_vis_n = 1

@pytest.mark.parametrize("batch_size", [1, 4])
def test_encoder_output_shape(batch_size):
    """
    Tests that the EncoderWrapper produces an output tensor
    of the correct shape.
    """
    img_size = 224
    opt = Opt()
    model = ViewDisentangleModel(opt)
    encoder = EncoderWrapper(model)
    
    # Create a dummy input tensor
    # Shape: [batch_size, channels, height, width]
    dummy_sketch = torch.randn(batch_size, 3, img_size, img_size)
    
    # Forward pass
    vertices, faces = encoder(dummy_sketch)
    
    # Check output shape
    expected_vertices_shape = (batch_size, opt.n_vertices, 3)
    assert vertices.shape == expected_vertices_shape, \
        f"Expected vertices shape {expected_vertices_shape}, but got {vertices.shape}"
