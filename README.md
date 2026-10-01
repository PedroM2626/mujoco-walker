# MuJoCo Walker Advanced RL Laboratory

This repository is a comprehensive **Reinforcement Learning and MLOps laboratory** dedicated to solving complex 3D bipedal locomotion, recovery from falls, and offline data utilization. We leverage MuJoCo physics alongside modern deep learning frameworks to benchmark state-of-the-art algorithms on Walker tasks.

Two custom environments carry the work: `WalkerRagdoll-v0` (`envs/walker_ragdoll_env.py`,
17 actuators, 46-dim state, 49 in the target phase) for Phases 1-3, and Gymnasium's
stock `Walker2d-v5` for the Phase-4 offline benchmark. There is no Humanoid environment in
this repo, despite earlier wording here.

---

## 🧠 Project Architecture & Phases

The repository is structured around several evolutionary phases of experimentation, encompassing everything from World Models and Ensembles to Model Merging and cutting-edge Offline-to-Online Reinforcement Learning.

### 🟢 Phase 1: Advanced Online RL Algorithms (Root Directory)
We implemented highly efficient online RL paradigms to train the robot from scratch:
- **DreamerV3 (`train_dreamer.py`):** An implementation of the World Models paradigm. The agent builds a latent hallucination of the MuJoCo physics engine to plan its walking steps internally before acting.
- **REDQ (`train_redq.py`):** *Randomized Ensembled Double Q-Learning*. Uses an aggressive ensemble of 10 Q-Networks and a high Update-To-Data (UTD) ratio to achieve massive sample efficiency on the Walker.
- **ARS (`train_ars.py`):** *Augmented Random Search*. A highly parallelized gradient-free evolutionary algorithm that searches for optimal linear policies in the parameter space.

⚠️ **No results are recorded for this phase.** These three are implementations, not
measurements: they produce no checkpoints under `checkpoints/` and no table in this file,
and they are not covered by the test suite. Two defects found while auditing them are now
fixed — `train_redq.py` updated its actor on every one of the 20 UTD iterations instead of
every `policy_frequency` gradient steps, and `train_dreamer.py` logged only
`obs[0]`/`actions[0]` while stepping 4 environments, discarding 75% of the data it
collected. Run them and record the output before citing numbers.

Phase 1's main trained model is the SAC/PPO/TD3 walker in `train_walker.py`, whose
checkpoints do exist (`checkpoints/walker_target_v1`, `walker_recovery_v1`).

### 🟡 Phase 2: MuJoCo MPC & Imitation Learning (Root Directory)
To achieve mathematically perfect locomotion, we tapped into the official DeepMind C++ MuJoCo MPC (Model Predictive Control) planner:
- We extracted **15,000 flawless transitions** of the MPC planner optimizing the walker's physics implicitly (`dataset.csv`).
- **Behavioral Cloning (`train_walker.py`):** Trained a PyTorch neural network to supervise-clone the MPC's optimal torque decisions, effectively caching the heavy MPC computation into a fast neural policy.

⚠️ **Not reproducible from this repository as committed.** `dataset.csv` is nowhere in the
tree (nor in git history), and the C++ MPC binary under `mujoco_mpc_walker/build/` is not
committed either, so there is nothing here to retrain the BC policy from. `verify.py` and
`openai_walker/train.py` both look for that file and exit with a clear error when it is
missing. The Phase-2 numbers below are therefore claims, not measurements you can check.
Reproduce by building `mujoco_mpc_walker` (`build.bat` / its Dockerfile), running the
planner to write `dataset.csv`, then `python openai_walker/train.py`.

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

#### 📊 Empirical Results (Phase 3 Benchmark)

| Merging Strategy | Avg Reward | Survival Rate | Analysis |
|:---|:---:|:---:|:---|
| **Hardcoded Supervisor** | **-11191.88** | *see caveat* | 🥇 Best reward of the four. The deterministic IF/ELSE baseline that swaps policies on torso height and upright factor. |
| **Weight Averaging (50/50)** | -11692.61 | *see caveat* | 🥈 Interpolating the parameter matrices kept the gait smoother than the arithmetic variants. |
| **Mixture of Experts (MoE)** | -11763.92 | *see caveat* | The learned router tracked the hardcoded supervisor closely. |
| **Task Arithmetic** | -16958.91 | *see caveat* | ⚠️ **Degradation.** Adding the isolated "recovery task vector" to the walking weights kept the robot controllable but heavily corrupted the gait. |

⚠️ **Two problems with the column this table was praised for.** The original run was
recorded as 1,000 episodes per model; no artifact of that run is in the repo, so the
reward numbers above cannot be re-derived from anything committed here. And the
"100% Survival Rate" row-by-row result was not a finding about robustness — it was a
property of the harness: `evaluate_merging.py` defined survival as "`env.terminated` never
became True", while these runs use `terminate_when_unhealthy=False`, so `terminated` is
`False` by construction and **no strategy could ever have been reported as dying**. That
script now seeds its resets, scores survival from the environment's own health condition,
reports falls per episode, and applies the `obs_rms` baked into the checkpoints (which it
previously ignored, meaning it had been driving a different policy than `play.py`). Re-run
it with `--num-episodes` to regenerate this table on a protocol that can actually
distinguish the strategies.

Also note that `play.py` resolves checkpoints only under `checkpoints/<run-id>/`, so the
root-level `merged_avg_model.pt`, `merged_ta_model.pt` and `moe_gate.pt` are not reachable
through its `--run-id` interface; use `evaluate_merging.py --avg-ckpt/--ta-ckpt/--gate-ckpt`
or move those artifacts under `checkpoints/`.

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
5.  **Inverse Reinforcement Learning (IRL):** `train_irl_gail.py` implements a Generative Adversarial Imitation Learning (GAIL) paradigm. It trains a Discriminator to distinguish expert from agent, and trains a SAC agent purely on the Discriminator's adversarial reward, completely bypassing the MuJoCo environment's native reward.
6.  **Grand Evaluation:** `play_race.py` sequentially simulates all generated models in the arena.

---

## 🏆 Final Benchmark Results (Offline-to-Online Race)

**These are the scores recorded in `openai_walker/final_results.txt`, the actual output of
`play_race.py`.** An earlier version of this README carried different numbers for several
rows (BC+SAC Regularized 4030.66, BC 3837.80, Teacher 3876.30, BC+SAC Constrained 203.54,
CQL+SAC 394.00, GAIL 1016.41) and called BC+SAC the overall champion. The tracked artifact
disagrees with all of that and puts BCQ first, so the table below now matches the evidence
in the repo. If you want the older figures back, they are in the git history — they are not
reproducible from anything committed here.

Read the caveat with the table: every score is **one episode** (`play_race.py` defaults to
`episodes=1`, reset unseeded). Walker2d returns are high-variance at n=1, so treat these as
identities ("this model runs at all") rather than measurements. A re-run with a fixed seed
and 50+ episodes is the honest version of this experiment and has not been done.

| Model Architecture | Final Score | Analysis |
|:---|:---:|:---|
| **Batch-Constrained Q-learning (BCQ)** | **3897.63** | 🏆 **Top score, and the best strictly-offline model.** It beat the online teacher without taking a single environment step during training, by using a VAE to propose actions and a perturbation network to refine them, which keeps it on the data manifold instead of exploiting unseen state-action pairs. |
| **Teacher (Online SAC)** | 3865.41 | The pure online expert that generated the dataset, and the upper bound the offline methods are measured against. |
| **BC+SAC (Regularized)** | 3396.81 | **Best offline-to-online hybrid.** Started from BC weights, then kept exploring with SAC while a BC loss regularised the actor to prevent catastrophic forgetting. It lands below BCQ and below the teacher in this recording, so it is not the champion an earlier version of this file claimed. |
| **Behavioral Cloning (BC)** | 1728.81 | Pure supervised cloning of the teacher. The dataset was narrow and deterministic, so cloning worked, but at roughly 45% of the teacher rather than the near-match previously reported here. |
| **Decision Transformer (DT)** | 1288.02 | 🧠 Recast Walker as sequence modelling with a causal transformer, conditioned on Return-To-Go. Scored this in only 10 epochs (100k steps) of training. |
| **Extra Trees Cloner (sklearn)** | *not in the recorded race* | 🌳 Trained in under 38 seconds on CPU and scored ~2522 when run separately (`openai_walker/eval_extratrees.py`, 5 episodes). It is **absent from `final_results.txt`**, so it is not ranked against the rest; it scores ~3900 in-distribution and drops to ~900 when forced to extrapolate, which is why the variance matters more than the mean. |
| **BC+SAC (Naive)** | 1369.87 | Unregularised: the SAC critic was random at first contact and its gradients overwrote the cloned policy early. Better than pure BC here, but well below the regularized variant. |
| **Inverse RL (GAIL)** | 997.57 | 🤖 Learned entirely from an adversarial discriminator's reward, with no knowledge of the environment reward. Trained to 1,000,000 steps and plateaued near 1000: the dataset was too deterministic, so the discriminator became a perfect judge and starved the actor of gradient. |
| **CQL+SAC** | 401.80 | Fine-tuning a Q-function that had already collapsed on the narrow dataset. |
| **BC+SAC (Constrained)** | 338.67 | Hard-clipping the actor's actions to the BC actions destroyed gradient propagation. |
| **CQL Offline** | 315.37 | Conservative Q-Learning failed to learn. Offline RL needs diverse, overlapping datasets for the Bellman backup to mean anything; on a single narrow expert path, Q-learning collapses. |
| **MaxEnt IRL** | 269.61 | Classic feature-expectation matching with a linear reward model ($r = \theta^T \phi$), too weak to express 3D bipedal locomotion. |
| **Inverse RL (AIRL)** | -6.36 | 💥 The first version produced `NaN` gradients where deterministic dataset actions hit the `atanh` limits. Rebuilt with spectral normalisation, action clipping ($\pm 0.95$) and an $h(s)$ reward-shaping baseline; still stuck near -5. A 20-hour server restart killed a long run mid-training. |
| **IQL Offline** | -15.81 | Implicit Q-Learning also collapsed for lack of dataset diversity. |

⚠️ **The environment this table needs is not the one in `.venv`.** Every Phase-4 script
makes `Walker2d-v5`, which only exists from **gymnasium 1.0**; the committed `.venv` is
Python 3.8 with gymnasium 0.29.1 (v4 at best) and has no `stable-baselines3`, so
`train_teacher.py`, `train_bc_sac*.py` and `play_race.py` fail at import there. Use
`requirements-phase4.txt` in a Python 3.10+ environment.

---

## ⚡ Simulation Throughput

`bench_env.py` measures the environment and splits the cost between MuJoCo physics and the
Python around it, so speedups can be A/B'd rather than asserted:

```bash
python bench_env.py --seconds 4                     # physics, single env, sync and parallel vec
python bench_env.py --mode parallel --n 32          # one backend only
python bench_env.py --sweep --n 32                  # envs-per-worker curve
python bench_env.py --json benchmarks/my_run.json   # machine-readable
```

Measured on an i9-14900HX (24 cores), mujoco 3.2.3, task_phase=recovery. Numbers are
env-steps/s, which is what a fixed `--total-timesteps` budget actually waits on:

| Configuration | env-steps/s | vs. committed |
|:---|---:|---:|
| Before (SyncVectorEnv, n=32) | 1,024 | — |
| Reward path rewritten (SyncVectorEnv, n=32) | 1,536 | **1.5x** |
| `--vec-backend parallel`, n=32 | 6,694 | **6.5x** |

Three things mattered, and one deliberate non-change:

1. **The reward path was doing string work per step.** `floor_contact_counts` looked up
   geom and body *names* for every contact, and `step()` counted contacts twice (once via
   `healthy_reward`). Geometry/body ids are now resolved once at construction and counted
   in a single pass; scalar `np.clip` / `np.linalg.norm` became builtins and `math.hypot`.
   Numerically equivalent: against the previous implementation over 4 task phases x 5 reset
   modes x 3 seeds, max `|obs diff|` 1.8e-15 and max `|reward diff|` 3.6e-13.
   `tests/test_env_golden.py` pins those rollouts so the MDP cannot drift unnoticed.
2. **`SyncVectorEnv` is a for-loop.** Stepping 32 envs in the trainer thread made n=32
   *slower than n=1* — the throughput was flat in `num_envs`. `envs/parallel_vector_env.py`
   runs worker processes and subclasses `SyncVectorEnv`, so autoreset, `_add_info` and
   observation concatenation are inherited rather than reimplemented;
   `tests/test_parallel_vec_env.py` proves the two backends agree step-for-step. Because
   each worker costs ~1s to start, `--vec-backend auto` only enables it from 16 envs up.
3. **`sparse_info`**: the ~30-key step info dict is only read on the step that ends an
   episode, so it is dropped from every other reply (~40% of parallel throughput).
   `--vec-dense-info` restores it.

**Physics was left alone on purpose.** `walker_ragdoll.xml` still compiles RK4 at
`timestep=0.002` with `frame_skip=5`. Euler measures 2.1x on physics-only, but diverges
from RK4 by `|dq| = 1.96` over 2,000 identical actions — enough that every checkpoint in
`checkpoints/` would be navigating a different MDP. The training budget (env steps) is
unchanged everywhere above; only wall clock moves.

### 🚫 JAX / MJX: measured, and it loses here

MJX (MuJoCo XLA) was evaluated as the obvious "make the env much faster" candidate, with
`mujoco-mjx==3.2.7` + `jax==0.10.2`, batched `vmap` + `jit`, `frame_skip=5` so an
"env step" means exactly what it means in `WalkerRagdollEnv.step`, compile time excluded.
`bench_mjx.py` reproduces this; raw results are in `benchmarks/mjx_cpu_batch.json` and
`benchmarks/mjx_gpu_wsl.json`.

env-steps/s (one policy action per world, higher is better):

| Path | env-steps/s | vs. ParallelVectorEnv |
|:---|---:|---:|
| **`ParallelVectorEnv` n=32, MuJoCo-C, CPU** | **6,694** | 1.00x |
| MJX-CPU n=32 (native Windows, jaxlib has no CUDA wheel) | 32 | 0.005x |
| MJX-CPU n=128 | 23 | 0.003x |
| MJX-GPU n=32 (WSL2, RTX 4070 Laptop 8 GB) | 191 | 0.03x |
| MJX-GPU n=512 | 1,263 | 0.19x |
| MJX-GPU n=1024 | 1,275 | 0.19x |
| MJX-GPU n=1024, **Euler + skip `rne_postconstraint`** | 4,338 | 0.65x |

MJX does not win on this robot, and the reason is structural rather than a tuning miss:

- **MJX is dense.** All 138 geom pairs of `walker_ragdoll.xml` are collision-tested and
  solved every step, so its cost barely depends on the state. MuJoCo-C measured `ncon=3`
  in the same fallen pose — roughly 46x fewer constraint rows to actually process.
- **RK4 costs about 4x in MJX** (935 → 3,828 env-steps/s when switching to Euler), and the
  relaxed configuration that gets to 4,338 changes the physics, so it is not comparable to
  the trained checkpoints in this repo.
- **CPU MJX gets worse with batch size**: one XLA CPU device splits the same throughput
  across worlds.
- `mjx.step` does not populate `cfrc_ext`, so the `impact_cost` term needs
  `mj_rnePostConstraint`, worth about another 24%.

Conclusion: the batched-GPU bet does not pay off for *this* model. Sticking with MuJoCo-C
across worker processes is both faster and keeps the exact dynamics the checkpoints were
trained under. MJX would start to compete on massively parallel quadruped/humanoid
scans with a single solver-friendly contact set, not on a 17-actuator ragdoll with an RK4
integrator. Native Windows GPU is unavailable for jaxlib regardless — the GPU numbers above
were taken under WSL2 with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.6` on the 8 GB card.

Also note `test_parallel_vec_env` skips on gymnasium >= 1.0: the parallel backend extends
`SyncVectorEnv.reset_wait/step_wait`, which gymnasium 1.0 removed. The trainers detect that
and fall back to `SyncVectorEnv` rather than failing.

---

## 🛠️ Technology Stack & MLOps

*   **Environments:** `mujoco` native bindings & `gymnasium` (Walker2d-v5)
*   **Algorithms:** PyTorch (IQL, CQL, BC, MoE, REDQ) & Stable-Baselines3 (SAC)
*   **MLOps Tracking:** `mlflow` (All offline experiments log metrics, losses, and artifact weights to `mlruns.db`)
*   **Containerization:** Docker with full OpenGL/OSMesa support for reproducible rendering.

---

## 🚀 Installation & Setup

The repository needs **two different interpreters**, because Phases 1-3 and Phase 4 pin
incompatible Gymnasium versions: `WalkerRagdoll-v0` and the trained checkpoints live with
gymnasium 0.29, while `Walker2d-v5` only exists from gymnasium 1.0. Installing one
requirements file into the other environment breaks it.

| Environment | Requirements | Phases | Verified with |
|:---|:---|:---|:---|
| `.venv` | `requirements.txt` | 1-3 (root scripts) | Python 3.8.10, gymnasium 0.29.1, mujoco 3.2.3, torch 2.4.1 |
| `.venv-phase4` | `requirements-phase4.txt` | 4 (`openai_walker/`) | Python 3.11.9, gymnasium 1.0.0, stable-baselines3 2.4.0, torch 2.5.1+cu121 |
| `.venv-mjx` | `requirements-mjx.txt` | MJX/JAX stepping | Python 3.11.9, mujoco 3.2.7, mujoco-mjx 3.2.7, jax 0.10.2 |

```bash
python -m venv .venv-phase4        # Python 3.10+ required
.venv-phase4\Scripts\activate      # Windows   (source .venv-phase4/bin/activate on Linux)
pip install -r requirements-phase4.txt
```

⚠️ `numpy<2` is required with stable-baselines3 2.4.0, and pip will happily install numpy 2.x
alongside it; SB3 then fails at import. Pin it explicitly if you hit that.

⚠️ **`torch` from PyPI is a CPU-only wheel on Windows.** The committed `.venv` has
`torch 2.4.1+cpu`, so every gradient update in the trainers runs on CPU even though the
machine has a GPU. The trainers already select `cuda` when it is available, so installing a
CUDA build is a pure speedup with no change to the training budget:
`pip install --index-url https://download.pytorch.org/whl/cu121 torch` (verified working on
this machine's RTX 4070 Laptop: `torch.cuda.is_available() -> True`).

### Option 1: Docker (recommended for rendering)
Build and run the Docker container to ensure all MuJoCo rendering libraries are pre-configured:
```bash
docker build -t mujoco-walker-rl .
docker run -it --rm mujoco-walker-rl bash
```
The image installs `requirements.txt` only, so it covers Phases 1-3; add
`requirements-phase4.txt` inside the container for the offline benchmark, and note that
GPU-accelerated MJX needs a Linux/CUDA container (`jaxlib` ships no CUDA wheels for
Windows).

### Option 2: Native Virtual Environment
```bash
python -m venv .venv
# Activate the environment
.venv\Scripts\activate   # (Windows)
source .venv/bin/activate # (Linux/Mac)

pip install -r requirements.txt
```

## ✅ Running the tests

The suite is plain `unittest` (no pytest required) and covers the environment contract, the
golden reward rollouts, the parallel/serial vector-env parity, checkpointing and the race
harness — 32 tests, ~3 min:

```bash
python -m unittest discover -s tests -t .
```

`.github/workflows/ci.yml` runs the same suite plus a Phase-4 import check on both
Gymnasium versions.

---

## 📖 Monitoring Experiments

All experiments across Phase 4 (Offline RL) are fully integrated with MLFlow.
To visualize the loss curves, Q-value estimations, and download the `.pt` artifacts:
```bash
# Run this from the root directory
mlflow ui --backend-store-uri sqlite:///mlruns.db
```
Navigate to `http://localhost:5000` in your browser.

The history is meant to live in **one** database, at the repository root. It used to be two:
11 Phase-4 scripts wrote `sqlite:///mlruns.db` and 5 wrote `sqlite:///../mlruns.db`, and
because `run_all.bat` runs from inside `openai_walker/`, half the runs landed in
`openai_walker/mlruns.db` while `mlflow ui` from the root showed only the other half.
`utils/mlflow_uri.py` (wrapped by `openai_walker/mlflow_backend.py`) now resolves the path
from the repository root, so the working directory no longer decides where a run goes. The
old `openai_walker/mlruns.db` still holds the runs that landed there — migrate it, or set
`MLFLOW_TRACKING_URI`, if you need that history in the same UI.

## 🔬 Reproducing and measuring

```bash
python -m unittest discover -s tests -t .   # 32 tests: env contract, golden rewards, vec parity
python bench_env.py --seconds 4             # env throughput, physics vs Python split
python bench_mjx.py --sizes 32,128          # MJX/JAX batched stepping
python verify.py                            # Phase-2 artifact check (exits 2 when missing)
```

`verify.py` currently exits non-zero because `dataset.csv` is not in the repository — see the
Phase-2 caveat. That is the accurate answer, not a defect in the checker.
