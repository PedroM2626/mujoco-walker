# MuJoCo Walker2d Offline RL Benchmark

This repository contains a full **Machine Learning Operations (MLOps)** pipeline designed to benchmark Offline Reinforcement Learning algorithms on the Gymnasium `Walker2d-v5` MuJoCo environment.

It transitions from a purely online Soft Actor-Critic (SAC) expert teacher to generating static datasets, followed by training pure offline models and finally executing Offline-to-Online fine-tuning.

## 🏆 Final Benchmark Results (The Offline Race)

We benchmarked 8 different models trained exclusively on 100,000 transitions collected from an expert SAC Teacher. The models were evaluated in a visual race (`play_race.py`) to measure their average return:

| Model Architecture | Final Score | Analysis |
|:---|:---:|:---|
| **BC+SAC (Regularized)** | **4030.66** | 🏆 **Champion.** By using the BC loss as an active regularizer, the SAC policy avoided catastrophic forgetting and successfully explored the environment to find a slightly more optimal gait, *beating the original teacher!* |
| **Teacher (Online SAC)** | 3876.30 | The pure online expert used to generate the dataset. |
| **Behavioral Cloning (BC)** | 3837.80 | Since the dataset was highly narrow and deterministic (only expert trajectories), pure supervised cloning was extremely effective and almost perfectly matched the teacher. |
| **BC+SAC (Naive)** | 972.04 | Without regularization, the SAC immediately destroyed the pre-trained BC weights in the first few epochs (Catastrophic Forgetting) because its untrained Critic started sending random gradients to the Actor. |
| **CQL+SAC** | 402.77 | Fine-tuning a collapsed Q-function yielded poor results. |
| **CQL Offline** | 315.96 | Conservative Q-Learning failed to learn. Why? Offline RL algorithms require *diverse* datasets with overlaps to properly backup Q-values. Given only a single, narrow expert path, Q-learning collapses. |
| **BC+SAC (Constrained)** | 146.17 | Hard-clipping the SAC actions to the BC actions destroyed the gradient propagation and ruined the policy. |
| **IQL Offline** | -17.37 | Implicit Q-Learning also collapsed due to the lack of dataset diversity (narrow expert data). |

---

## 🛠️ Technology Stack & MLOps

*   **Environment:** `gymnasium` (Walker2d-v5 backend by MuJoCo)
*   **Algorithms:** PyTorch (Custom neural networks for IQL, CQL, BC) & Stable-Baselines3 (Teacher SAC)
*   **MLOps Tracking:** `mlflow` (All offline experiments log metrics, losses, and artifact weights to `mlruns.db`)
*   **Containerization:** Docker for reproducible environments

## 📂 Project Structure

All execution code lives in the `openai_walker/` directory.

```text
D:\mujoco-walker\
├── openai_walker/
│   ├── train_teacher.py               # Trains the online expert SAC
│   ├── generate_dataset.py            # Rolls out the expert to create dataset_openai.csv
│   ├── train_bc.py                    # Pure Behavioral Cloning
│   ├── train_iql.py                   # Pure Implicit Q-Learning
│   ├── train_cql.py                   # Pure Conservative Q-Learning
│   ├── train_bc_sac_*.py              # 3 variations of Offline-to-Online Fine-Tuning for BC
│   ├── train_iql_sac.py               # Offline-to-Online Fine-Tuning for IQL
│   ├── train_cql_sac.py               # Offline-to-Online Fine-Tuning for CQL
│   └── play_race.py                   # Visual MuJoCo evaluation script
├── mlruns.db                          # MLFlow SQLite Database
├── requirements.txt                   # Exact pip dependencies
└── Dockerfile                         # Container definition
```

## 🚀 Installation & Setup

### Option 1: Docker (Recommended)
Build and run the Docker container to ensure all MuJoCo rendering libraries are pre-configured:
```bash
docker build -t mujoco-walker-rl .
docker run -it --rm mujoco-walker-rl bash
```

### Option 2: Native Virtual Environment
```bash
python -m venv .venv
# Activate the environment
.venv\Scripts\activate   # (Windows)
source .venv/bin/activate # (Linux/Mac)

pip install -r requirements.txt
```

---

## 📖 Step-by-Step Execution Guide

To reproduce the entire benchmark from scratch, navigate to the `openai_walker/` directory and execute the pipeline sequentially:

### 1. Train the Online Expert (Teacher)
Train a pure Soft Actor-Critic agent from scratch to reach expert status (~3800 points).
```bash
cd openai_walker
python train_teacher.py
```
*(This will save `sac_walker2d_final.zip`)*

### 2. Generate the Offline Dataset
Extract 100,000 state-action-reward transitions from the Teacher and save them to a CSV.
```bash
python generate_dataset.py
```
*(This will generate `dataset_openai.csv`)*

### 3. Pure Offline RL Training
Train the completely offline paradigms using the CSV dataset. All runs are automatically tracked in MLFlow.
```bash
python train_bc.py
python train_iql.py
python train_cql.py
```

### 4. Offline-to-Online Fine-Tuning
Load the pre-trained weights from the offline phase and fine-tune them dynamically in the environment using SAC variants.
```bash
# Behavioral Cloning Fine-Tuning Variants
python train_bc_sac.py
python train_bc_sac_regularized.py
python train_bc_sac_constrained.py

# IQL and CQL Fine-Tuning Variants
python train_iql_sac.py
python train_cql_sac.py
```

### 5. Final Evaluation (The Race)
Run the visualization script to load all 8 generated models and pit them against each other in a rendered MuJoCo GUI.
```bash
python play_race.py
```

### 6. MLOps Monitoring
You can monitor all training losses, rewards, and artifact states using the MLflow UI:
```bash
# Run this from the root directory
mlflow ui --backend-store-uri sqlite:///mlruns.db
```
Then navigate to `http://localhost:5000` in your browser.
