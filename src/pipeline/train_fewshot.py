import argparse
import yaml
import pandas as pd
from tqdm import tqdm
import torch

from ..utils.config import load_config
from ..utils.seed import set_seed
from ..dataio.sketches import SketchDataset
from ..sketch2model_core.models.view_disentangle_model import ViewDisentangleModel
from ..sketch2model_core.wrappers import EncoderWrapper, DecoderWrapper
from ..sketch2model_core.models.base_model import BaseModel
from ..sketch2model_core.utils.utils import find_class_using_name
from ..fewshot_core.adapter import PerClassAdapter
from ..fewshot_core.scorer import FewShotScorer
from ..fewshot_core.losses import get_loss_function
from ..pipeline.episodic_sampler import EpisodicSampler
from ..fewshot_core.classes import CLASS_TO_ID

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


def run_episode(episode: dict, scorer: FewShotScorer, device: torch.device):
    """
    Runs a single few-shot episode, calibrating the scorer and
    evaluating performance on the query set.
    """
    support_sketches, support_labels = episode['support_set']
    query_sketches, query_labels = episode['query_set']
    
    # Move to device
    support_sketches = support_sketches.to(device)
    support_labels = support_labels.to(device)
    query_sketches = query_sketches.to(device)
    query_labels = query_labels.to(device)

    # Calibrate the scorer (fits adapter and normalizes losses)
    scorer.calibrate(support_sketches, support_labels)

    correct = 0
    total = 0
    
    # Evaluate on query set
    for i in range(len(query_sketches)):
        sketch = query_sketches[i].unsqueeze(0) # Add batch dimension
        true_label_id = query_labels[i].item()
        
        predicted_class_name, _ = scorer.predict(sketch)
        predicted_class_id = CLASS_TO_ID[predicted_class_name] # Assuming CLASS_TO_ID is accessible
        
        if predicted_class_id == true_label_id:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


def main():
    parser = argparse.ArgumentParser(description="Run Few-Shot Learning Training and Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to the main configuration file.")
    parser.add_argument("--paths", type=str, default="configs/paths.yaml", help="Path to the data/weights paths configuration file.")
    parser.add_argument("--eval_only", action="store_true", help="If set, runs evaluation only.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on ('cpu' or 'cuda').")
    args = parser.parse_args()

    # --- Setup ---
    set_seed(42)
    device = torch.device(args.device)
    
    cfg = load_config(args.config)
    paths_cfg = load_config(args.paths) # Load paths config
    
    mode = "EVALUATION" if args.eval_only else "TRAINING"
    print(f"--- Running in {mode} mode ---")

    # --- Models ---
    opt = Opt()
    model = ViewDisentangleModel(opt)
    model.load_networks(opt)
    
    encoder = EncoderWrapper(model)
    decoder = DecoderWrapper(model)

    adapter = PerClassAdapter(
        latent_dim=cfg['latent_dim'], 
        num_classes=13, 
        hidden_dim=cfg['adapter']['hidden']
    ).to(device)

    # --- Data ---
    dataset = SketchDataset(root_dir=paths_cfg['data']['sketches_root'], img_size=cfg['data']['img_size'])
    sampler = EpisodicSampler(
        dataset,
        n_way=cfg['N_way'],
        k_shot=cfg['K_shot'],
        q_query=cfg['Q_query'],
        num_episodes=cfg['episodes']
    )
    
    # --- Scorer and Loss ---
    loss_fn = get_loss_function(cfg['loss']['type'])
    scorer = FewShotScorer(
        encoder, 
        decoder, 
        loss_fn,
        adapter,
        normalize=cfg['loss']['normalize_per_class']
    )

    # --- Main Loop ---
    episode_results = []
    total_correct = 0
    total_query = 0

    for i, episode in enumerate(tqdm(sampler, desc=f"Running Episodes")):
        accuracy, correct, total = run_episode(episode, scorer, device)
        episode_results.append({
            "episode": i,
            "accuracy": accuracy,
            "n_way": cfg['N_way'],
            "k_shot": cfg['K_shot']
        })
        total_correct += correct
        total_query += total
    
    # --- Logging ---
    results_df = pd.DataFrame(episode_results)
    mean_accuracy = total_correct / total_query if total_query > 0 else 0.0
    
    print("\n--- Results ---")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    
    # Save results to a CSV file
    import os
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "evaluation_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Saved evaluation results to {results_path}")

if __name__ == "__main__":
    main()
