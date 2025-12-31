import argparse
import torch
from PIL import Image
from torchvision import transforms

from ..sketch2model_core.models.view_disentangle_model import ViewDisentangleModel
from ..sketch2model_core.wrappers import EncoderWrapper, DecoderWrapper
from ..sketch2model_core.models.base_model import BaseModel
from ..sketch2model_core.utils.utils import find_class_using_name
from ..utils.config import load_config
from ..fewshot_core.adapter import PerClassAdapter
from ..fewshot_core.scorer import FewShotScorer
from ..fewshot_core.losses import get_loss_function

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
        self.template_path = 'sketch2model-fewshot/src/sketch2model_core/SoftRas/data/obj/sphere/sphere_642.obj'
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

def main():
    parser = argparse.ArgumentParser(description="Run inference on a single sketch.")
    parser.add_argument("--config", type=str, required=True, help="Path to the main configuration file.")
    parser.add_argument("--paths", type=str, default="configs/paths.yaml", help="Path to the data/weights paths configuration file.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input sketch image.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on ('cpu' or 'cuda').")
    args = parser.parse_args()

    # --- Setup ---
    device = torch.device(args.device)
    cfg = load_config(args.config)
    
    print(f"--- Running Inference on {args.image} ---")

    # --- Models ---
    opt = Opt()
    model = ViewDisentangleModel(opt)
    model.load_networks(opt)
    
    encoder = EncoderWrapper(model)
    decoder = DecoderWrapper(model)
    # For inference, adapter is not trained, it acts as identity
    adapter = PerClassAdapter(latent_dim=cfg['latent_dim'], num_classes=13).to(device)

    encoder.eval()
    decoder.eval()
    adapter.eval()

    # --- Scorer and Loss ---
    # Scorer doesn't need calibration/normalization for single-image inference.
    loss_fn = get_loss_function(cfg['loss']['type'])
    scorer = FewShotScorer(encoder, decoder, loss_fn, adapter, normalize=False)

    # --- Image Preprocessing ---
    transform = transforms.Compose([
        transforms.Resize((cfg['data']['img_size'], cfg['data']['img_size'])),
        transforms.Grayscale(),
        transforms.ToTensor(),
        lambda x: 1 - x, # Invert colors (black sketch on white background to white on black)
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    try:
        image = Image.open(args.image).convert("L")
        sketch_tensor = transform(image).unsqueeze(0).to(device)
    except FileNotFoundError:
        print(f"ERROR: Image file not found at {args.image}")
        return

    # --- Inference ---
    pred_class, losses = scorer.predict(sketch_tensor)

    # --- Display Results ---
    print("\n--- Reconstruction Losses per Class ---")
    sorted_losses = sorted(losses.items(), key=lambda item: item[1])
    for class_name, loss in sorted_losses:
        print(f"  - {class_name:<15}: {loss:.6f}")
        
    print(f"\n---> Predicted Class: {pred_class}")

if __name__ == "__main__":
    main()
