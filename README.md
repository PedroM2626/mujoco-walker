# MuJoCo Walker Advanced RL Laboratory

This repository is a comprehensive **Reinforcement Learning and MLOps laboratory** dedicated to solving complex 3D bipedal locomotion, recovery from falls, and offline data utilization. We leverage MuJoCo physics alongside modern deep learning frameworks to benchmark state-of-the-art algorithms on Humanoid/Walker tasks.

---

## 🧠 Project Architecture & Phases

The repository is structured around several evolutionary phases of experimentation, encompassing everything from World Models and Ensembles to Model Merging and cutting-edge Offline-to-Online Reinforcement Learning.

### 🟢 Phase 1: Advanced Online RL Algorithms (Root Directory)
We implemented highly efficient online RL paradigms to train the robot from scratch:
- **DreamerV3 (`train_dreamer.py`):** An implementation of the World Models paradigm. The agent builds a latent hallucination of the MuJoCo physics engine to plan its walking steps internally before acting.
- **REDQ (`train_redq.py`):** *Randomized Ensembled Double Q-Learning*. Uses an aggressive ensemble of 10 Q-Networks and a high Update-To-Data (UTD) ratio to achieve massive sample efficiency on the Walker.
- **ARS (`train_ars.py`):** *Augmented Random Search*. A highly parallelized gradient-free evolutionary algorithm that searches for optimal linear policies in the parameter space.

### 🟡 Phase 2: MuJoCo MPC & Imitation Learning (Root Directory)
To achieve mathematically perfect locomotion, we tapped into the official DeepMind C++ MuJoCo MPC (Model Predictive Control) planner:
- We extracted **15,000 flawless transitions** of the MPC planner optimizing the walker's physics implicitly (`dataset.csv`).
- **Behavioral Cloning (`train_walker.py`):** Trained a PyTorch neural network to supervise-clone the MPC's optimal torque decisions, effectively caching the heavy MPC computation into a fast neural policy.

### 🟠 Phase 3: Transfer Learning, Model Merging & Mixture of Experts (Root Directory)
How do we combine a "Walking Policy" with a "Fall Recovery Policy" without catastrophic forgetting? To achieve this, we utilized a strict **Transfer Learning Curriculum** and advanced Model Merging techniques.

**The Transfer Learning Curriculum (Linear Mode Connectivity):**
To merge two different neural networks, they must share the same *Linear Mode Connectivity Basin*. If two networks are trained from different random initializations, averaging their weights produces garbage. 
1. First, we pre-trained a base agent to walk perfectly (`walker_target_v1` - 40M steps).
2. Then, we performed **Transfer Learning**: we duplicated these pre-trained weights and spawned a new training environment focused *exclusively* on recovering from extreme falls (`walker_recovery_v1` - 20M steps).
3. Because the recovery agent was fine-tuned from the walking agent, they share the same geometric parameter space, allowing us to perform algebraic operations on their matrices.

**Merging Techniques Evaluated:**
- **Task Arithmetic (`merge_models.py`):** Subtracts the base walking weights from the fine-tuned recovery weights to isolate a pure "Recovery Task Vector" ($\tau = \theta_{rec} - \theta_{walk}$). This vector is then added algebraically to any policy.
- **Weight Averaging (`merge_models.py`):** Directly interpolates the parameter matrices of the two policies ($\theta_{avg} = 0.5 \cdot \theta_{rec} + 0.5 \cdot \theta_{walk}$).
- **Mixture of Experts / MoE Gate (`train_moe_gate.py`):** A routing network (`moe_gate.pt`) trained to dynamically switch the robot's control between the Walking Policy and the Recovery Policy based on its current pitch/velocity (e.g., if it detects a fall, it activates the recovery expert).
- **Evaluation (`evaluate_merging.py` & `play.py`):** Scripts to visually inspect how well the merged/MoE models transition between walking and standing up.

#### 📊 Empirical Results (Phase 3 Benchmark - 1000 Episodes)
The models were evaluated under extreme conditions where the robot is subjected to forces that induce falling. The benchmark was massively scaled to **1,000 episodes** per model (4,000,000 steps total) to measure true long-term robustness. The goal is to survive (stand back up) while maintaining the highest possible reward (least penalty).

| Merging Strategy | Avg Reward | Survival Rate | Analysis |
|:---|:---:|:---:|:---|
| **Hardcoded Supervisor** | **-11191.88** | 100% | 🥇 **Upper Bound.** The deterministic IF/ELSE baseline that manually swaps policies. |
| **Weight Averaging (50/50)** | -11692.61 | 100% | 🥈 **Highly Robust.** Counter to intuition, interpolating the weights created an incredibly robust hybrid policy that survived 1000 extreme episodes without a single fatal failure, smoothing out the transitions between walking and recovering. |
| **Mixture of Experts (MoE)** | -11763.92 | 100% | The neural router successfully learned to mimic the hardcoded supervisor dynamically, matching its survival rate flawlessly over 1000 episodes. |
| **Task Arithmetic** | -16958.91 | 100% | ⚠️ **Degradation.** The algebraic addition of the "recovery task vector" to the walking weights successfully kept the robot alive (100% survival), but heavily corrupted the walking gait, resulting in severe penalties. |

### 🔴 Phase 4: Offline RL & Offline-to-Online Benchmarking (`openai_walker/` folder)
We migrated to the standardized `Walker2d-v5` Gymnasium environment to conduct a massive benchmark on learning *strictly from static datasets*, without querying the environment.

1.  **Online Expert Generation:** `train_teacher.py` trains a flawless Soft Actor-Critic (SAC) model.
2.  **Dataset Mining:** `generate_dataset.py` records 100k transitions from the teacher (`dataset_openai.csv`).
3.  **Pure Offline Training:** 
    - `train_bc.py`: Supervised Imitation.
    - `train_iql.py`: Implicit Q-Learning (Expectile Regression to avoid out-of-distribution queries).
    - `train_cql.py`: Conservative Q-Learning (Penalizes Q-values for unseen actions).
4.  **Offline-to-Online Fine-Tuning:** Injecting offline weights into an active SAC for continued environmental exploration:
    - `train_bc_sac.py` (Naive initialization)
    - `train_bc_sac_regularized.py` (BC Loss Penalty in the Actor)
    - `train_bc_sac_constrained.py` (Action Constraints / Clipping)
    - `train_iql_sac.py` & `train_cql_sac.py`
5.  **Grand Evaluation:** `play_race.py` sequentially simulates all generated models.

---

## 🏆 Final Benchmark Results (Offline-to-Online Race)

Below are the empirical results from running `play_race.py` on the 100k expert transitions dataset:

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

*   **Environments:** `mujoco` native bindings & `gymnasium` (Walker2d-v5)
*   **Algorithms:** PyTorch (IQL, CQL, BC, MoE, REDQ) & Stable-Baselines3 (SAC)
*   **MLOps Tracking:** `mlflow` (All offline experiments log metrics, losses, and artifact weights to `mlruns.db`)
*   **Containerization:** Docker with full OpenGL/OSMesa support for reproducible rendering.

---

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

## 📖 Monitoring Experiments

All experiments across Phase 4 (Offline RL) are fully integrated with MLFlow.
To visualize the loss curves, Q-value estimations, and download the `.pt` artifacts:
```bash
# Run this from the root directory
mlflow ui --backend-store-uri sqlite:///mlruns.db
```
Navigate to `http://localhost:5000` in your browser.
