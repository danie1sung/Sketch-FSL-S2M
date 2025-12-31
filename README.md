# `sketch2model-fewshot`

A minimal, runnable Python repository for few-shot sketch-based 3D object classification.

## Goal

This project provides a framework for classifying free-hand sketches into a fixed set of 13 3D object classes. It is designed for few-shot learning scenarios, where the model can adapt to new variations with only a few examples.

## Method

The core of the classification is a reconstruction-based approach. For a given input sketch:
1. The sketch is encoded into a 3D latent vector.
2. For each of the 13 possible classes, the model attempts to decode the latent vector into a 3D object of that class.
3. A reconstruction loss (e.g., silhouette difference) is computed for each attempted class.
4. The class that yields the minimum reconstruction loss is chosen as the prediction. This is the **argmin-loss rule**.

This approach avoids direct classification and instead leverages the generative capabilities of the underlying Sketch2Model architecture, which is now integrated directly from the original `sketch2model` repository.

## Fixed Classes

The model is trained and evaluated on the following 13 fixed classes:
- Airplane
- Bench
- Cabinet
- Car
- Chair
- Display
- Lamp
- Loudspeaker
- Rifle
- Sofa
- Table
- Telephone
- Watercraft

## Quickstart

1.  **Installation**

    First, ensure you have a C++ compiler (like MSVC on Windows or GCC on Linux) and the NVIDIA CUDA Toolkit installed on your system. The CUDA Toolkit is free to download from NVIDIA's official website.

    Then, install the package in editable mode along with its dependencies:
    ```bash
    pip install -e .
    ```

    Additionally, the integrated `SoftRas` module requires compilation. Navigate to its directory and install it:
    ```bash
    cd src/sketch2model_core/SoftRas
    python setup.py install
    cd ../../../
    ```
    This will install `soft_renderer` into your Python environment.

2.  **Configuration**

    - `configs/fewshot.yaml`: Contains parameters for few-shot learning episodes (N-way, K-shot), model dimensions, and loss settings.
    - `configs/paths.yaml`: **You must edit this file.** Specify the root directory for your sketch dataset. The model weights are directly loaded within the code from the `checkpoints` directory, specifically configured for the `chair_pretrained` model as an example. You may need to adjust the `template_path` in `src/pipeline/train_fewshot.py` and `src/pipeline/infer.py` if you use a different template mesh.

3.  **Run Scripts**

    - **Train Few-Shot Adapter:**
      ```bash
      ./scripts/run_train.sh
      ```
      This script runs the few-shot training process to calibrate per-class adapters using the settings in `configs/fewshot.yaml`.

    - **Evaluate:**
      ```bash
      ./scripts/run_eval.sh
      ```
      This runs the evaluation on a test set of episodes.

    - **Inference:**
      ```bash
      ./scripts/run_infer.sh path/to/your/sketch.png
      ```
      This script classifies a single sketch image.

## Model Integration

This repository now directly integrates the 3D encoding and decoding logic from the `sketch2model` repository into `src/sketch2model_core`. The previous `src/sketch2model_io` module has been removed.

The `ViewDisentangleModel` from `sketch2model_core` is used for both encoding and decoding. Pre-trained weights are expected to be available in the `checkpoints` directory. The current implementation defaults to loading the `chair_pretrained` model.

## Evaluation

The evaluation protocol follows an N-way K-shot episodic format. The main metric is top-1 classification accuracy.

Results, including per-episode performance and summary statistics, are saved to a CSV file in the `outputs/` directory (created automatically).