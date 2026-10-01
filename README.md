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

**Status: implementations that now have recorded, explicitly short evidence runs.** Until
2026-10-01 none of the three had ever been run in-repo: no checkpoints, no numbers, no test
coverage. Two defects were found and fixed while checking them — `train_redq.py` updated its
actor on every one of the 20 UTD iterations instead of every `policy_frequency` gradient
steps, and `train_dreamer.py` recorded only `obs[0]`/`actions[0]` while stepping 4
environments, throwing away three quarters of the transitions it simulated.

Short runs (minutes, not the 1M-step budget) are recorded in `benchmarks/phase1_evidence.json`
and reproducible with `python summarize_phase1.py`:

| Trainer | Budget | Episodes | Mean return | Max return |
|:---|---:|---:|---:|---:|
| `train_ars.py` | 60,384 steps | 101 eval cycles | eval peak **11116.19** at epoch 23, 3087.62 at the end | 11116.19 |
| `train_redq.py` | 6,000 steps | 184 | 2951.29 | 10800.01 |
| `train_dreamer.py` | 8,000 steps | 351 | 2041.18 | 7205.90 |

These say "the algorithm runs and collects reward on the ragdoll", nothing more. Do not
quote them as results: the budgets are 0.6% of the intended 1M steps, and ARS in particular
peaked early and decayed, which at this length says nothing about the method.

Phase 1's properly trained models are the SAC walkers in `checkpoints/walker_target_v1`
(40M steps) and `checkpoints/walker_recovery_v1` (20M steps), evaluated in Phase 3 above.

### 🟡 Phase 2: MuJoCo MPC & Imitation Learning (Root Directory)
To achieve mathematically perfect locomotion, we tapped into the official DeepMind C++ MuJoCo MPC (Model Predictive Control) planner:
- We extracted **15,000 flawless transitions** of the MPC planner optimizing the walker's physics implicitly (`dataset.csv`).
- **Behavioral Cloning (`train_walker.py`):** Trained a PyTorch neural network to supervise-clone the MPC's optimal torque decisions, effectively caching the heavy MPC computation into a fast neural policy.

⚠️ **Not reproducible as committed, for a specific and fixable reason.** `dataset.csv` is
nowhere in the tree or in git history, and the C++ collector that would produce it is
**disabled in the source**: `mujoco_mpc_walker/main.cc` contains a complete transition
writer (`my_step_callback`, writing
`target_x,target_y,qpos_*,qvel_*,ctrl_*,reward,done`) but the six lines in `main()` that
open the file and install it as `mjcb_sensor` are commented out (~lines 100-105), so the
prebuilt `walker_mpc.exe` opens the interactive MJPC GUI and writes nothing. Regenerating
therefore needs: uncomment that block, a C++ toolchain (Visual Studio Build Tools + CMake at
the paths `build.bat` hard-codes — not installed on this machine), a rebuild, and a manual
GUI collection session. `verify.py` prints exactly these steps and exits 2 until the
artifacts exist. The versioned, reproducible alternative is Phase 4's
`openai_walker/dataset_openai.csv` (100k SAC-teacher transitions), which every table in this
file that quotes a number actually measured against.

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

Re-measured 2026-10-01 with the corrected harness: 20 episodes per strategy, seeded resets
(`--seed 7`, episode *i* uses `7+i`), each expert fed through the `obs_rms` baked into its
own checkpoint, and survival read from the environment's health condition instead of from
`terminated` (which these runs never set). Evidence: `eval_phase3_20ep.log` in the run
directory, summary in `benchmarks/phase3_merging_20ep.json`, recomputed with
`python summarize_benchmarks.py`.

| Merging Strategy | Mean Reward | Median | Std | Falls / episode | Ended standing |
|:---|---:|---:|---:|---:|---:|
| **Hardcoded Supervisor** | **8585.55** | **-1307.99** | 35860.97 | 1.00 | 5% |
| **Mixture of Experts (MoE)** | 2741.24 | -4833.61 | 27946.03 | 0.75 | 5% |
| **Task Arithmetic** | -18781.05 | -14303.47 | 10502.82 | 0.20 | 0% |
| **Weight Averaging (50/50)** | -20290.83 | -13344.22 | 12482.88 | 0.10 | 0% |

Read this table before citing it:

1. **The spread dwarfs the mean.** Std is 3-15x the mean and every median is negative: the
   supervisor's +8585 comes from a single 128,298-point episode out of 20. Ranking by mean
   over 20 episodes of this task is not a stable measurement.
2. **The previously published ordering was a bug, not a finding.** The retired table put
   Weight Averaging second ("highly robust, counter to intuition") and MoE third, on
   un-normalised observations. Once each policy gets the inputs it was trained on, Weight
   Averaging is **last** and MoE is second. That reversal is the point of fixing the harness.
3. **"100% survival" was never observable.** The old script defined survival as
   "`env.terminated` never became True", while these runs use
   `terminate_when_unhealthy=False` — so `terminated` is False by construction and no
   strategy could ever have been reported as dying. Real fall counts are in the table: the
   merged policies fall least because they do almost nothing, and finish upright 0% of the
   time.
4. The supervisor is the intended upper bound and still leads on both mean and median; MoE
   tracks it with fewer falls. Task Arithmetic remains clearly degraded.

To regenerate: `python evaluate_merging.py --num-episodes 20 --seed 7` (the old artifacts
were unreachable from `play.py`; it now takes `--checkpoint` for the root-level merged
models and `--moe --gate/--recovery/--target` for the router, which the README had claimed
for a script containing no MoE code).

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

Headline numbers are the **20 seeded episodes** below, re-measured on 2026-10-01; the
single-episode record that shipped with the repo is kept underneath as history, because its
per-model explanations are still the substance of this phase.

An earlier version of this README quoted numbers that no artifact in the repo supports
(BC+SAC Regularized 4030.66, BC 3837.80, Teacher 3876.30, BC+SAC Constrained 203.54, CQL+SAC
394.00, GAIL 1016.41) and crowned BC+SAC the champion. Both the older figures and the
"champion" claim are wrong against `openai_walker/final_results.txt`, and the
single-episode file itself turned out to be a poor estimator once re-run with a protocol
that can average.

### Measured again, properly: 20 seeded episodes per model (2026-10-01)

The single-episode record above is only an identity check, and that turned out to matter a
lot. Re-run with `evaluate_all.py --episodes 20 --seed 123` (seeded per episode:
episode *i* resets with `123+i`), the ranking and the magnitudes both move. Evidence:
`openai_walker/final_results_20ep_seed123.txt`, `benchmarks/phase4_race_20ep.json`,
recomputed with `python summarize_benchmarks.py`.

| Model | Mean (20 ep) | Std | Min | Max | Reading |
|:---|---:|---:|---:|---:|:---|
| **Teacher (Online SAC)** | **3762.80** | 420.00 | 1938.74 | 3943.94 | Upper bound, and the only model whose score is stable. |
| **Behavioral Cloning (BC)** | 3681.22 | 493.74 | 2307.43 | 4275.67 | **The single-episode record badly understated it** (1728.81). Cloning the teacher is nearly as good as the teacher; the n=1 draw caught a bad episode. |
| **Extra Trees Cloner (sklearn)** | 3302.68 | 869.89 | 1248.99 | 4055.37 | Third, and now measured on the same protocol instead of a separate 5-episode run. Huge spread confirms the in-distribution/extrapolation split. Loaded with sklearn 1.5.2 although trained with 1.3.2 — treat as indicative. |
| **Batch-Constrained Q-learning (BCQ)** | 3004.64 | 962.27 | 854.52 | 4152.96 | Best *strictly* offline model, but its min of 854 shows one unlucky episode costs a full rank. |
| **BC+SAC (Regularized)** | 2718.70 | 918.77 | 1363.63 | 4004.96 | The hybrid finishes below plain BC on this protocol: fine-tuning on top of cloning did not pay for itself here. |
| **Decision Transformer (DT)** | 1977.32 | 1083.90 | 961.23 | 3700.57 | Better than its n=1 score, with the widest spread in the table. |
| **BC+SAC (Naive)** | 1118.61 | 514.70 | 533.66 | 2378.10 | Catastrophic forgetting, as described: unregularised SAC overwrote the cloned policy. |
| **Inverse RL (GAIL)** | 997.97 | 0.44 | 997.09 | 998.67 | Almost zero variance — this policy has converged onto a fixed, mediocre gait. |
| **CQL+SAC** | 407.44 | 5.66 | 399.79 | 416.48 | Fine-tuning a Q-function that already collapsed on the narrow dataset. |
| **CQL Offline** | 322.58 | 8.17 | 309.61 | 346.75 | Offline RL needs diverse, overlapping data for the Bellman backup to mean anything. |
| **MaxEnt IRL** | 260.81 | 40.32 | 189.05 | 357.04 | Linear reward model ($r = \theta^T \phi$) too weak for bipedal locomotion. |
| **BC+SAC (Constrained)** | 126.70 | 79.90 | -1.58 | 233.96 | Hard-clipping actions to the BC policy destroyed gradient flow. |
| **Inverse RL (AIRL)** | -6.31 | 0.05 | -6.41 | -6.22 | `NaN` gradients from deterministic actions at the `atanh` limits; rebuilt with spectral norm + clipping + $h(s)$ shaping and still flat. |
| **IQL Offline** | -11.51 | 15.42 | -18.15 | 54.83 | Collapsed for lack of dataset diversity. |

What this re-measurement changes, stated plainly:

1. **The teacher is first, not BCQ.** At n=1 BCQ looked like the champion; over 20 seeded
   episodes the online SAC teacher is ahead of every offline method, which is the expected
   result for an upper bound and the sanity check the old table failed.
2. **Behavioral Cloning was misreported by a factor of two** in the single-episode record,
   and it beats every offline *value-based* method and the BC+SAC hybrids.
3. **Variance dominates ranking at n=20 for the offline methods** (std 500-1100 vs
   differences of similar size between neighbours). Re-run with more episodes before
   quoting any pairwise comparison.
4. **PQR is missing, not slow:** `train_irl_pqr.py` exists but no PQR artifact was ever
   produced, so it is silently absent from both records.

### Historical record: one unseeded episode per model (`final_results.txt`)

Kept because the per-model *explanations* below are the substance of this phase, and
because this is the file the earlier version of this README contradicted. The scores are
single unseeded draws — compare them with the 20-episode table above rather than quoting
them.

| Model Architecture | Final Score | Analysis |
|:---|:---:|:---|
| **Batch-Constrained Q-learning (BCQ)** | **3897.63** | 🏆 **Best strictly-offline model.** It beat the online teacher without taking a single environment step during training, by using a VAE to propose actions and a perturbation network to refine them, which keeps it on the data manifold instead of exploiting unseen state-action pairs. |
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

⚠️ **How to read these numbers.** This is a laptop CPU whose clocks vary with power and
thermal state, and repeat runs of the identical command have ranged ~2x apart (the sync
n=32 configuration measured 1,408, 1,536 and 2,720 env-steps/s in three runs the same
minute-scale window apart). The absolute column is therefore indicative; the **ratios** are
what to trust, because each was measured back-to-back in one process against the same
baseline. Re-measure on your own hardware with `python bench_env.py --seconds 4` before
quoting a multiplier.

What the rewrite actually bought is visible in the last line `bench_env.py` prints: the
Python around MuJoCo in one `env.step` fell from ~46% of wall clock to **7%** (23 µs of a
312 µs step, physics 289 µs), which is why the remaining headroom is in parallelism and
in the integrator rather than in Python.

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

⚡ **`torch` from PyPI is a CPU-only wheel on Windows**, so with the original `.venv`
(`torch 2.4.1+cpu`) every gradient update ran on CPU on a machine with an RTX 4070. The
`.venv` here now has `torch 2.4.1+cu121`. Measured at an identical 20k-step SAC budget
(8 envs, sync backend, updates from step 2000, same seed):

| Device | Wall clock | Throughput |
|:---|---:|---:|
| `--device cpu` | 105 s | 190 env-steps/s |
| `--device cuda` | 30 s | 666 env-steps/s |

That is **3.5x end-to-end** at this budget, and 2.4x on the isolated update step (11.28 ms
→ 4.75 ms per critic+actor pair at batch 512, `bench_device.py`). The
gain shrinks as `num_envs` grows, because rollout collection becomes the dominant cost —
which is why the env work above matters more than the device.

Use `--device cpu|cuda|auto` to choose. Do **not** force CPU with
`CUDA_VISIBLE_DEVICES=-1`: with a CUDA build of torch that segfaults partway through
training here (reproduced twice at 20k steps with updates; the env-only path does not
crash, and the in-process flag does not either).

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

## 🧹 Repository size

`.git` was **1.2 GB** while the real history is only ~28 MB. The difference was a single
**unreachable** 1.15 GB object: `openai_walker/extratrees_model.pkl` had been `git add`ed
(force-added past the `*.pkl` ignore rule) on 2026-06-28 and later reset, leaving the blob
dangling — no commit ever referenced it, so `git rev-list --objects --all` did not show it
and GitHub never received it. `git prune --expire=now && git gc --prune=now` removed it and
`git fsck` is clean; the history is untouched and the clone is now small.

Two facts to keep in mind:

- `openai_walker/extratrees_model.pkl` (2.24 GB on disk) is **not** version-controlled and
  is regenerable with `python train_extratrees.py`. Do not force-add it.
- `openai_walker/dataset_openai.csv` (74 MB) **is** tracked, on purpose: it is the only copy
  of the Phase-4 dataset, and every table here is measured against it. If the repo ever
  needs to shed it, migrate with `git lfs migrate import --include=...` and a force-push -
  that rewrites published history, so it needs every clone to re-fetch.

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
python bench_device.py                        # SAC update cost, CPU vs CUDA
```

`verify.py` currently exits non-zero because `dataset.csv` is not in the repository — see the
Phase-2 caveat. That is the accurate answer, not a defect in the checker.
