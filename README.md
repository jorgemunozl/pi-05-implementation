# LeRobot VLA Experiments: Pi0.5 & Testing Suite

This repository contains implementations, experiments, and automated tests for Vision-Language-Action (VLA) models, specifically focusing on the **Pi0.5** model released by Physical Intelligence.

The code utilizes the Hugging Face [LeRobot](https://github.com/huggingface/lerobot) library to run inference, fine-tuning, and evaluation benchmarks. All experiments are provided as Jupyter Notebooks, which are automatically tested using `nbval` to ensure reproducibility.

## 📌 Features

*   **Pi0.5 Implementation**: Notebooks demonstrating how to load and run inference with the `pi0.5` model using LeRobot.
*   **Automated Notebook Testing**: A CI/CD-ready testing suite that uses `pytest` and `nbval` to validate notebook execution.
*   **Evaluation Benchmarks**: Experiments running zero-shot evaluation on the Libero benchmark.
*   **Custom Training**: Examples of fine-tuning Pi0.5 on custom datasets (e.g., Aloha or custom LeRobot datasets).

## 🛠 Installation

### Prerequisites
*   Python 3.10+
*   CUDA-enabled GPU (Recommended for VLA inference)

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/lerobot-pi05-experiments.git
cd lerobot-pi05-experiments
```

### 2. Create a Virtual Environment
```bash
conda create -n lerobot-pi05 python=3.10
conda activate lerobot-pi05
```

### 3. Install LeRobot with Pi0 Support
You need to install `lerobot` with the specific extras required for Physical Intelligence models.

```bash
# Install lerobot from source (recommended for latest model support)
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[pi]"
cd ..

# Install testing dependencies
pip install pytest nbval ipykernel
```

> **Note**: Accessing Pi0.5 weights may require accepting the license on the [Hugging Face Hub](https://huggingface.co/physical-intelligence/pi0). Ensure you are logged in via `huggingface-cli login`.

## 📂 Repository Structure

```
lerobot-pi05-experiments/
├── notebooks/
│   ├── 01_pi05_inference.ipynb       # Basic inference demo with Pi0.5
│   ├── 02_libero_eval.ipynb          # Evaluation on Libero benchmark
│   └── 03_finetune_aloha.ipynb       # Fine-tuning experiment on Aloha dataset
├── tests/
│   └── conftest.py                   # Pytest configuration for notebooks
├── requirements.txt
└── README.md
```

## 🧪 Running Tests

We use `nbval` (Notebook Validation) to treat our Jupyter Notebooks as unit tests. This ensures that the code inside the notebooks actually runs and produces the expected output.

### Run All Notebook Tests
To test all notebooks in the repository:
```bash
pytest --nbval notebooks/
```

### Run a Specific Notebook Test
```bash
pytest --nbval notebooks/01_pi05_inference.ipynb
```

### How it works
*   **Strict Mode**: By default, `nbval` compares the output generated during testing with the output saved in the `.ipynb` file. If they differ, the test fails.
*   **Lax Mode**: If you only want to check that the cells execute without errors (ignoring specific output differences like timestamps or download bars), use:
    ```bash
    pytest --nbval-lax notebooks/
    ```

## 🔬 Experiments & Usage

### Experiment 1: Pi0.5 Inference (Zero-Shot)
*   **Notebook**: `notebooks/01_pi05_inference.ipynb`
*   **Description**: Loads the `physical-intelligence/pi0.5` policy. It downloads a sample episode from a LeRobot dataset and runs the model to generate action chunks (velocity/position) based on visual observations.

### Experiment 2: Libero Benchmark Evaluation
*   **Notebook**: `notebooks/02_libero_eval.ipynb`
*   **Description**: Sets up the Libero simulation environment and runs the Pi0.5 model against standard tasks (e.g., `libero_spatial`, `libero_object`).
*   **Metric**: Success rate over 20 evaluation episodes.

### Experiment 3: Fine-tuning on Custom Data
*   **Notebook**: `notebooks/03_finetune_aloha.ipynb`
*   **Description**: Demonstrates how to fine-tune the VLA model on the Aloha mobile dataset using `lerobot`'s training script. Includes configuration for:
    *   `optimizer`: AdamW
    *   `batch_size`: 8
    *   `steps`: 2000

## 📝 Troubleshooting

*   **Flash Attention Error**: Pi0 models often use Flash Attention 2. If you encounter errors, ensure you have it installed correctly for your CUDA version:
    ```bash
    pip install flash-attn --no-build-isolation
    ```
*   **OOM (Out of Memory)**: Pi0.5 is a large model. If running on a consumer GPU (e.g., RTX 3090/4090), try reducing the batch size or using `torch.float16`.
*   **Missing 'pi' Extra**: If you get imports errors related to `openpi`, ensure you installed lerobot with `pip install -e ".[pi]"`.

## 🤝 Contributing

1.  Fork the repo.
2.  Create your feature branch (`git checkout -b feature/amazing-experiment`).
3.  Add your notebook and **ensure it passes `pytest --nbval`**.
4.  Commit your changes.
5.  Open a Pull Request.

## 📜 License

This project is licensed under the Apache 2.0 License. Note that the Pi0.5 model weights are subject to their own license by Physical Intelligence.