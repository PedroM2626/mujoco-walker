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
| `train_ars.py` | 60,384 steps | 101 eval cycles | no trend; range 1521-6378 across epochs | 11116.19 |
| `train_redq.py` | 6,000 steps | 184 | 2951.29 | 10800.01 |
| `train_dreamer.py` | 8,000 steps | 351 | 2041.18 | 7205.90 |

These say "the algorithm runs and collects reward on the ragdoll", nothing more. Do not
quote them as results: the budgets are 0.6% of the intended 1M steps. ARS shows no learning
trend across its 101 epochs - evaluation returns go 853.74 to 11116.19, mean 3602.54, std
1790.17, correlation with epoch -0.09, and the fitted drift over the whole run is 0.31 std,
i.e. inside the noise of a single-rollout evaluation. An earlier draft of this file described
that as "peaked early and decayed", which was reading noise as signal. What the numbers do
say: return tracks episode length (corr 0.77) and lengths stay at 13-119 steps, so the linear
policy is surviving briefly rather than walking.

Phase 1's properly trained models are the SAC walkers in `checkpoints/walker_target_v1`
(40M steps) and `checkpoints/walker_recovery_v1` (20M steps), evaluated in Phase 3 above.

The scale of that table matters: REDQ and Dreamer wrap the environment in `NormalizeReward`,
so the `Mean return` column is a normalised sum, not the reward the raw environment reports.
On the raw scale the same checkpoints are still indistinguishable from an inert robot -
measured over an identical seeded 300-step rollout of `WalkerRagdoll-v0`, REDQ scores
-3826.77, Dreamer -3836.79 and ARS -3887.55, against **-3775.76 for commanding zero** and
-3857.06 for uniform random actions. `tests/test_phase1_behaviour.py` pins that reading: it
checks the loading contract of each evidence checkpoint (recorded observation width, matching
`obs_rms`, actions inside the action box, `deterministic=True` really being deterministic) and
asserts that the 6k-step actor is *not* better than doing nothing, so the claim breaks the
moment someone reads these rows as behaviour.

### 🟡 Phase 2: MuJoCo MPC & Imitation Learning (Root Directory)
To achieve mathematically perfect locomotion, we tapped into the official DeepMind C++ MuJoCo MPC (Model Predictive Control) planner:
- We extracted **15,000 flawless transitions** of the MPC planner optimizing the walker's physics implicitly (`dataset.csv`).
- **Behavioral Cloning (`train_walker.py`):** Trained a PyTorch neural network to supervise-clone the MPC's optimal torque decisions, effectively caching the heavy MPC computation into a fast neural policy.

⚠️ **Not reproducible as committed, and the blocker was two layers deeper than it looked.**
`dataset.csv` is nowhere in the tree or in git history. The obvious cause is real:
`mujoco_mpc_walker/main.cc` contains a complete transition writer (`my_step_callback`, writing
`target_x,target_y,qpos_*,qvel_*,ctrl_*,reward,done`) but the lines that installed it were
commented out, so the June binary in `build/walker_mpc.exe` opens the GUI and writes nothing.
Those lines are now flags: `--dataset_path=dataset.csv --transitions=15000` records, and with
no flag the binary behaves exactly as it always did.

Underneath that, **the CMake project could not configure at all.** `CMakeLists.txt` declared
`add_executable(mjpc_dataset_tool generate_data.cc walker_task.cc)`, and `generate_data.cc` is
not in the repository and never was committed, so `cmake ..` aborts with "Cannot find source
file: generate_data.cc" whatever compiler is installed. The stale target is gone; `build/`
still holds the `mjpc_dataset_tool.dir` object folder from the day the file existed locally,
which is how it was traced. `build.bat` then hardcoded `C:\Program Files (x86)\Microsoft
Visual Studio\18\...` and `C:\Program Files\CMake`, neither of which exists on this machine, so
it now resolves the toolchain through `vswhere`.

What remains is a system install, not a code fix: no MSVC, no CMake, no compiler of any kind is
present here (`where cl / gcc / cmake` all empty), so none of the above has been compiled, and
collection is still a manual GUI session. A headless collector would need `mjpc::Agent` wired
by hand - `Initialize` → `Allocate` → `Reset` → `SetState` → `PlanIteration` →
`ActionFromPolicy` - which cannot be built or tested on this machine. Installing *Desktop
development with C++* makes `build.bat` runnable; the versioned, measured alternative remains
Phase 4's `openai_walker/dataset_openai.csv` (100k SAC-teacher transitions), which every table
in this file that quotes a number actually measured against. `verify.py` prints the same steps
and exits 2 until the artifacts exist.

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

Re-measured 2026-10-01 with the corrected harness, **100 episodes per strategy**, seeded
resets (`--seed 11`, episode *i* uses `11+i`), each expert fed through the `obs_rms` baked
into its own checkpoint, and survival read from the environment's health condition instead
of from `terminated` (which these runs never set). A 20-episode run at seed 7 gave the same
ordering. Evidence: `benchmarks/phase3_merging_100ep.json`, recomputed with
`python summarize_benchmarks.py`.

| Merging Strategy | Mean Reward | Median | Std | Falls / episode | Ended standing |
|:---|---:|---:|---:|---:|---:|
| **Hardcoded Supervisor** | **13549.68** | **831.31** | 31259.57 | 1.31 | 1% |
| **Mixture of Experts (MoE)** | 9199.76 | -4391.54 | 32424.26 | 0.92 | 2% |
| **Task Arithmetic** | -17410.69 | -15126.37 | 8661.12 | 0.24 | 0% |
| **Weight Averaging (50/50)** | -18313.84 | -13297.76 | 10650.28 | 0.17 | 0% |

Read this table before citing it:

1. **The mean is carried by outliers.** The supervisor's std (31259.57) is 2.3x its mean and
   MoE's (32424.26) is 3.5x, while only the supervisor's median is positive (831.31) against
   MoE's -4391.54. Delete the single best episode and the supervisor
   falls from 13549.68 to 12239.87, MoE from 9199.76 to 8042.06; delete the best five and they
   become 8889.50 and 4437.68. 32% of supervisor episodes score above +20k while 28% score
   below -10k, so this is a bimodal task: rank by median, or say which one you mean.
2. **The previously published ordering was one specific line of code, and it has now been
   reproduced rather than asserted.** The retired table (`git show 32da152:README.md`) read
   -11191.88 / -11692.61 / -11763.92 / -16958.91 for supervisor / weight averaging / MoE /
   task arithmetic, which ranked Weight Averaging second ("highly robust, counter to
   intuition"). The cause was in `evaluate_merging.py`: it called
   `single_agent.get_action(obs_49)` on raw observations and never applied the `obs_rms`
   pickled inside each checkpoint. Re-running the *corrected* script with `--raw-obs` - which
   now exists solely to reproduce that input path - at the same 100 episodes and seed 11:

   | Merging strategy | corrected (`obs_rms`) | raw obs = old harness | retired publication |
   |:---|---:|---:|---:|
   | Hardcoded Supervisor | 13549.68 | -11691.90 | -11191.88 |
   | Weight Averaging (50/50) | -18313.84 | -11591.29 | -11692.61 |
   | Mixture of Experts | 9199.76 | -11782.31 | -11763.92 |
   | Task Arithmetic | -17410.69 | -17166.98 | -16958.91 |

   Every raw-obs value lands within 501 points of the number that was published from it, on a
   table whose whole spread is 5,767 points, and Task Arithmetic stays last in both. The old
   table is that bug, measured; evidence in `benchmarks/phase3_merging_100ep_rawobs.json`.
3. **Why that one line reorders rather than merely rescales: the input error is not uniform
   across the four paradigms.** On 200 real observations from a seeded rollout, feeding raw
   instead of normalised observations changes the deterministic action by:

   | Network | mean &#124;Δaction&#124; on a [-1,1] scale | correlation of raw vs correct |
   |:---|---:|---:|
   | recovery expert (46 dims) | 0.726 | 0.18 |
   | target expert (49 dims) | 0.613 | 0.39 |
   | weight-averaged merge | 0.219 | 0.86 |
   | task-arithmetic merge | 0.510 | 0.54 |

   The MoE *gate* agreed with itself 96.0% of steps either way, so the router was not the
   problem - its experts were. The two paradigms that depend on the experts (supervisor, MoE)
   were being driven as different policies and sank into the same band as the merged ones;
   weight averaging and task arithmetic, bad under both input paths, kept their place. That is
   how a benchmark where all four score about -11k becomes one where two score +9k to +13k.
4. **"100% survival" was never observable.** The old script defined survival as
   "`env.terminated` never became True", while these runs use
   `terminate_when_unhealthy=False`, so `terminated` is False by construction and no
   strategy could ever have been reported as dying. Real falls are 1.31 per episode for the
   supervisor and 0.17 for Weight Averaging - the merged policies fall least because they
   do almost nothing, and finish upright 0% of the time. The same signature shows up in the
   raw arm: its best supervisor episode scores 23,529 against 143,220 for the corrected one,
   because a raw-observation supervisor never gets a long walk at all (1% of episodes above
   +20k, against 32%).
5. MoE tracks the supervisor with fewer falls (0.92 against 1.31); Task Arithmetic stays
   clearly degraded in every arm.

To regenerate: `python evaluate_merging.py --num-episodes 100 --seed 11` (add `--raw-obs` for
the reproduction arm), then `python summarize_benchmarks.py`. The old artifacts were
unreachable from `play.py`; it now takes `--checkpoint` for the root-level merged models and
`--moe --gate/--recovery/--target` for the router, which the README had claimed for a script
containing no MoE code.

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

### Measured properly: 50 seeded episodes per model (2026-10-01)

Re-run with `evaluate_all.py --episodes 50 --seed 2026` (episode *i* resets with `2026+i`).
Evidence: `openai_walker/final_results_50ep_seed2026.txt`, the per-episode returns in
`openai_walker/final_episodes_50ep_seed2026.json`, and `benchmarks/phase4_race_50ep.json`; a
20-episode run at seed 123 gave the same ordering, so the picture is stable. Regenerate with
`python summarize_benchmarks.py`.

Because episode *i* is the same initial state for every model, neighbouring rows can be
compared as a **paired** sample instead of two means with error bars: `python paired_stats.py`
reads those per-episode arrays and reports the paired difference, a bootstrap interval,
paired-t and Wilcoxon p-values. Two of the three pairs people care about are ties (below), and
that is now a tested statement rather than an inference from the std.

| Model | Mean (50 ep) | Std | Min | Max | Reading |
|:---|---:|---:|---:|---:|:---|
| **Behavioral Cloning (BC)** | **3529.44** | 649.73 | 1638.67 | 4085.63 | Statistically tied with the teacher: paired over the same 50 seeded episodes the gap is +12.49 with a 95% CI of [-231.49, +265.05] (paired-t p=0.92). Cloning the expert recovers essentially all of it. |
| **Teacher (Online SAC)** | 3516.95 | 724.62 | 1511.13 | 4011.47 | The upper bound - and indistinguishable from BC. |
| **Extra Trees Cloner (sklearn)** | 3092.22 | 1034.38 | 953.80 | 3992.76 | Third, on the same 50-episode seeded protocol as every row above (seed 2026; trained in 6.3 s). The nesting shows how far to trust it: the first 20 episodes of *this* run average 3362.4146, and two separate fresh 20-episode runs at the same seed returned 3362.4230 and 3362.4146 - same model, same resets, 0.008 apart in the mean and up to 0.17 on one episode. The cause is measured: `Walker2d-v5` rollouts under a fixed action are bit-identical across trials, but this model's `predict` on the **same input** differs between repeated calls by 5.6e-16 (2.8e-16 for a freshly fit 100-tree forest with the same settings), and the chaotic simulator amplifies that. It is `n_jobs=-1`'s threaded reduction, not the sklearn version gap: a forest fit with `n_jobs=1` reproduces bit-exactly (0.00e+00 over repeated calls and uneven batch splits), so `train_extratrees.py` now uses `n_jobs=1`. The rows here were measured against the artifact as it sits on disk (2026-06-28, `n_jobs=-1`), so trust them to ~1e-5 relative, not to the second decimal. What moved the row's *level* is sample size: episodes 21-50 average 2912.10. |
| **BC+SAC (Regularized)** | 2784.29 | 816.62 | 1259.66 | 4019.63 | The offline-to-online hybrid finishes **below** plain BC: fine-tuning on top of cloning did not pay for itself. |
| **Batch-Constrained Q-learning (BCQ)** | 2732.60 | 1066.23 | 1273.22 | 4049.40 | Best strictly-offline method that is not plain imitation. This row moved: it was 2838.27 before `evaluate_all.py` started seeding the policy-side RNG, and BCQ draws its action through a sampled VAE, so the same protocol used to return 2838.27 and 2715.22 on two consecutive runs. It is now 5th, one place behind BC+SAC (Regularized) by 51.69 - a difference this protocol cannot resolve either way. |
| **Decision Transformer (DT)** | 1927.23 | 1101.18 | 933.58 | 3700.39 | Widest spread in the table; conditioned on Return-To-Go, 10 epochs of training. |
| **BC+SAC (Naive)** | 1290.57 | 476.50 | 516.50 | 2502.59 | Unregularised: the fresh critic's gradients overwrite the cloned policy. |
| **Inverse RL (GAIL)** | 998.07 | 0.37 | 997.31 | 998.74 | Near-zero variance - converged onto a fixed, mediocre gait; the discriminator starves the actor. |
| **CQL+SAC** | 407.99 | 5.93 | 395.03 | 424.70 | Fine-tuning a Q-function that already collapsed on the narrow dataset. |
| **CQL Offline** | 321.01 | 7.10 | 310.32 | 339.21 | Needs diverse, overlapping data for the Bellman backup to mean anything. |
| **MaxEnt IRL** | 279.33 | 34.47 | 197.38 | 354.70 | Linear reward model ($r = 	heta^T \phi$) too weak for bipedal locomotion. |
| **BC+SAC (Constrained)** | 146.65 | 112.59 | -1.66 | 542.12 | Hard-clipping actions to the BC policy destroyed gradient flow. |
| **IQL Offline** | 4.44 | 124.42 | -18.23 | 874.28 | Near zero on average, but a max of 874 - "collapsed" is the mean's story, not every episode's. |
| **Inverse RL (AIRL)** | -6.33 | 0.05 | -6.47 | -6.19 | Flat, and deterministic about being flat. |

What this measurement changes, stated plainly:

1. **BC and the teacher are tied**, so the defensible claim is "imitation recovers the
   expert", not "an offline method beat the teacher". At n=1 BCQ looked like the winner; at
   n=50 it is clearly behind both.
2. **The single-episode record misreported BC by about 2x** (1728.81 against 3529.44). It was
   one unlucky draw, and every other n=1 number inherits that risk. Unlike Phase 3 there is no
   wrong-input path behind this: `evaluate_all.py` applied each model's `scaler_*.pkl` then and
   applies it now, so the shift is sample size alone - CQL (315.37 → 321.01) and BC+SAC Naive
   (1369.87 → 1290.57) barely moved.
3. **The two close gaps are ties, and that is now tested instead of eyeballed.** Comparing the
   models episode-by-episode (same `seed+i` reset for every model) instead of mean-against-mean:
   BC − Teacher = **+12.49**, 95% bootstrap interval [-231.49, +265.05], paired-t p=0.92,
   Wilcoxon p=0.28, dz=0.014. BCQ − BC+SAC (Regularized) = **-51.69**, interval
   [-419.11, +316.80], p=0.79 on both tests. Reproduce with `python paired_stats.py
   --episodes 50 --seed 2026`; written to `benchmarks/phase4_paired_50ep_seed2026.json`.
4. **Extra Trees belongs in the table, not in a footnote**, and it beats BCQ only on the rank
   test: paired, Extra Trees − BCQ = +359.62 with interval [-49.05, +742.47] and paired-t
   p=0.084, while Wilcoxon gives p=0.0248. Read that as "it wins most episodes, but BCQ has
   enough good ones that the mean difference is not significant at 95%" - third place is real
   but not firm. Its min/max of 953.80/3992.76 is the in-distribution vs extrapolation split the
   retired "~2522, memorised the manifold" line was hiding.
5. **Making the table reproducible found a real defect.** `evaluate_all.py` seeded the
   environment but never the policy RNG, and BCQ acts through a sampled VAE
   (`z = mean + std * torch.randn_like(std)`), so BCQ was not a measurement of BCQ's policy - it
   was a draw from a distribution. Two consecutive 50-episode runs at the same seed returned
   2838.27 and 2715.22 while the other twelve models reproduced digit for digit. After seeding
   `torch` and `numpy` per model, two runs return the identical 2477.24 at n=3, the twelve
   unchanged rows still match their old values exactly, and BCQ's row is now 2732.60 - which
   drops it behind BC+SAC (Regularized) into 5th. The GUI reference `play_race.py`, whose
   unseeded single-episode output is `final_results.txt`, is seeded the same way.
6. **PQR is absent because it was never saved, not because it is slow.** `train_irl_pqr.py`
   exists and an MLflow run `Deep_PQR_IRL` is in the database with 0 metrics - the process
   died before writing `pqr_policy.pt`. `iql_sac_model.pt` is missing the same way, after 653
   metrics. Both were closed as stale (`closed_as_stale` tag) rather than deleted: the metric
   history is the only surviving evidence that they ran.

### Historical record: one unseeded episode per model (`final_results.txt`)

Kept because the per-model *explanations* below are the substance of this phase, and
because this is the file the earlier version of this README contradicted. The scores are
single unseeded draws — compare them with the 50-episode table above rather than quoting them.

| Model Architecture | Final Score | Analysis |
|:---|:---:|:---|
| **Batch-Constrained Q-learning (BCQ)** | **3897.63** | 🏆 **Best strictly-offline model.** It beat the online teacher without taking a single environment step during training, by using a VAE to propose actions and a perturbation network to refine them, which keeps it on the data manifold instead of exploiting unseen state-action pairs. |
| **Teacher (Online SAC)** | 3865.41 | The pure online expert that generated the dataset, and the upper bound the offline methods are measured against. |
| **BC+SAC (Regularized)** | 3396.81 | **Best offline-to-online hybrid.** Started from BC weights, then kept exploring with SAC while a BC loss regularised the actor to prevent catastrophic forgetting. It lands below BCQ and below the teacher in this recording, so it is not the champion an earlier version of this file claimed. |
| **Behavioral Cloning (BC)** | 1728.81 | Pure supervised cloning of the teacher. The dataset was narrow and deterministic, so cloning worked, but at roughly 45% of the teacher rather than the near-match previously reported here. |
| **Decision Transformer (DT)** | 1288.02 | 🧠 Recast Walker as sequence modelling with a causal transformer, conditioned on Return-To-Go. Scored this in only 10 epochs (100k steps) of training. |
| **Extra Trees Cloner (sklearn)** | *not in the recorded race* | 🌳 Trained in seconds on CPU and genuinely competitive. It is absent from `final_results.txt`, so it was never ranked against the rest; the "~2522 over 5 episodes / ~3900 in-distribution / ~900 when forced to extrapolate" breakdown quoted here exists nowhere in the repo as a measurement. On the 50-episode seeded protocol it measures 3092.22 ± 1034.38, which is third in the table above. |
| **BC+SAC (Naive)** | 1369.87 | Unregularised: the SAC critic was random at first contact and its gradients overwrote the cloned policy early. Better than pure BC here, but well below the regularized variant. |
| **Inverse RL (GAIL)** | 997.57 | 🤖 Learned entirely from an adversarial discriminator's reward, with no knowledge of the environment reward. Trained to 1,000,000 steps and plateaued near 1000: the dataset was too deterministic, so the discriminator became a perfect judge and starved the actor of gradient. |
| **CQL+SAC** | 401.80 | Fine-tuning a Q-function that had already collapsed on the narrow dataset. |
| **BC+SAC (Constrained)** | 338.67 | Hard-clipping the actor's actions to the BC actions destroyed gradient propagation. |
| **CQL Offline** | 315.37 | Conservative Q-Learning failed to learn. Offline RL needs diverse, overlapping datasets for the Bellman backup to mean anything; on a single narrow expert path, Q-learning collapses. |
| **MaxEnt IRL** | 269.61 | Classic feature-expectation matching with a linear reward model ($r = \theta^T \phi$), too weak to express 3D bipedal locomotion. |
| **Inverse RL (AIRL)** | -6.36 | 💥 The first version produced `NaN` gradients where deterministic dataset actions hit the `atanh` limits. Rebuilt with spectral normalisation, action clipping ($\pm 0.95$) and an $h(s)$ reward-shaping baseline; still stuck near -5. A 20-hour server restart killed a long run mid-training. |
| **IQL Offline** | -15.81 | Implicit Q-Learning also collapsed for lack of dataset diversity. |

⚠️ **The environment this table needs is not the one in `.venv`.** Every Phase-4 script
makes `Walker2d-v5`, which only exists from **gymnasium 1.0**; `.venv` is gymnasium 0.29.1
(v4 at best) and has no `stable-baselines3`, so `train_teacher.py`, `train_bc_sac*.py` and
`play_race.py` fail at import there. Use `requirements-phase4.txt` in a Python 3.10+
environment (`.venv-phase4` here).

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

`Dockerfile.mjx` is the third route: an `nvidia/cuda` base so these numbers can be reproduced
on any Linux host with the NVIDIA Container Toolkit, without WSL. It is **written but not
executed** - this machine has no Docker daemon (`docker: command not found`) - so unlike the
WSL2 and native-CPU rows above, nothing in it is a measurement. `docker build -f Dockerfile.mjx -t mujoco-walker-mjx .` then `docker run --rm --gpus all mujoco-walker-mjx` prints `jax.devices()`.

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
| `.venv` | `requirements.txt` | 1-3 (root scripts) | Python 3.11.9, gymnasium 0.29.1, mujoco 3.2.3, torch 2.4.1+cu121, numpy 1.26.4, mlflow 3.16.1 |
| `.venv-phase4` | `requirements-phase4.txt` | 4 (`openai_walker/`) | Python 3.11.9, gymnasium 1.0.0, stable-baselines3 2.4.0, torch 2.5.1+cu121 |
| `.venv-mjx` | `requirements-mjx.txt` | MJX/JAX stepping | Python 3.11.9, mujoco 3.2.7, mujoco-mjx 3.2.7, jax 0.10.2 |

```bash
python -m venv .venv-phase4        # Python 3.10+ required
.venv-phase4\Scripts\activate      # Windows   (source .venv-phase4/bin/activate on Linux)
pip install -r requirements-phase4.txt
```

⚠️ stable-baselines3 2.4.0 **declares** `numpy<2.0`, and pip prints a dependency conflict if
you install numpy 2.x next to it. Honoring the bound is the safe read; note that the import
itself did not fail when tested here (`numpy==2.1.3` + SB3 2.4.0 imported, warning only), so
the declared range is narrower than what breaks in practice - which is why this file pins
`numpy>=1.26.4` without inventing an upper bound of its own.

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
`requirements-phase4.txt` inside the container for the offline benchmark. GPU-accelerated MJX
needs a Linux/CUDA container (`jaxlib` ships no CUDA wheels for native Windows) - that is
`Dockerfile.mjx`, built with `docker build -f Dockerfile.mjx -t mujoco-walker-mjx .` and run
with `--gpus all`.

### Option 2: Native Virtual Environment
```bash
python -m venv .venv
# Activate the environment
.venv\Scripts\activate   # (Windows)
source .venv/bin/activate # (Linux/Mac)

pip install -r requirements.txt
```

⚠️ **Python 3.10 or newer.** `requirements.txt` asks for `mlflow>=3.0`, and mlflow 3.x has no
Python 3.8/3.9 distribution: on 3.8 pip answers `No matching distribution found`, and the
same file also needs `numpy>=1.26.4`, which 1.25 dropped 3.8 for. The repository's own history
is in the mlflow-3 schema, so a 3.8 environment can run the trainers but cannot log to it.

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

Everything under `checkpoints/` is local and ignored, so it never affected the clone size.
The working directory here was 31 GB, of which 12.8 GB was `checkpoints/walker_recovery_v1`:
twenty hourly checkpoints from the same 20M-step run, and only two of them are referenced by
any script - `sac_ckpt_20000000.pt` (the recovery expert in every Phase-3 number) and
`sac_ckpt_1000000.pt` (`merge_models.py --base-ckpt`, the transfer-learning starting point).
The other eighteen were deleted on 2026-10-01, freeing 11.4 GB; `checkpoints/` went from
12,853 MB to 1,454 MB, and the Phase-3 evaluation reproduces its four strategy means exactly
afterwards (36496.42 / 31237.52 / -8015.85 / -30245.87 at two episodes, seed 11), so the
surviving set is sufficient. `checkpoints/Humanoid_Curriculum_v1` (3.8 MB, four files,
referenced by nothing) went with it: it was the residue of a Humanoid claim that this README
no longer makes. What is *not* recoverable this way is the deleted steps - regenerating them
means re-running the 20M-step recovery training.

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

**Version requirement: mlflow 3.x, which needs Python 3.10+.** `mlruns.db` is at schema
revision `b7e2c1a4d9f3`, which only mlflow 3.x understands. With the `mlflow 2.17.2` that the
old Python 3.8 `.venv` shipped, opening it raises
`alembic.util.exc.CommandError: Can't locate revision identified by 'b7e2c1a4d9f3'` - and
because the trainers' mlflow helper is deliberately fault-tolerant, that error was swallowed,
so runs appeared to log while writing nothing. `requirements.txt` pins `mlflow>=3.0` for that
reason. (Found while adding the stale-run tool, which had to open the same database.)

mlflow 3.x publishes no Python 3.8 distribution - `pip download --no-deps "mlflow>=3.0"`
returns `No matching distribution found` - so the fix had to be the interpreter, not a
`pip install`. `.venv` is now Python 3.11.9 with mlflow 3.16.1, and the logging path was
checked end-to-end: `start_mlflow_run` → `log_mlflow_metrics` → `end_mlflow_run` against a
throwaway sqlite backend produced a FINISHED run carrying both metrics and the `seed` param.
The swap is numerically invisible: the same 200-step rollout from seed 7 gives reward
-2506.5429333387096 in the 3.8 and 3.11 environments, and `evaluate_merging.py --num-episodes
2 --seed 11` reproduces all four strategy means to the last printed digit. The old
environment is kept as `.venv-py38-backup/` (two `mv` commands to go back), and
`start_mlflow_run` still prints the version and the remedy when it hits the wall, so a
freshly-cloned 3.8 venv fails loudly instead of silently.

The history now lives in **one** database, at the repository root: 21 runs, 5 experiments,
252,856 metric rows. It used to be two: 11 Phase-4 scripts wrote `sqlite:///mlruns.db` and 5
wrote `sqlite:///../mlruns.db`, and because `run_all.bat` runs from inside `openai_walker/`,
half the runs landed in `openai_walker/mlruns.db` while `mlflow ui` from the root showed only
the other half. `utils/mlflow_uri.py` (wrapped by `openai_walker/mlflow_backend.py`) now
resolves the path from the repository root, so the working directory no longer decides where
a run goes. The 13 runs that had already landed in the wrong file were merged in with
`python openai_walker/mlflow_backend.py merge --source sqlite:///.../openai_walker/mlruns.db
--apply`; the source file is kept untouched as `openai_walker/mlruns.merged-into-root.db`,
and the merge is keyed on `run_uuid`, so re-running it moves nothing (verified: `0 new run(s)`).

Three runs were left in RUNNING by killed processes - `AIRL_IRL` (74,297 metric rows),
`Deep_PQR_IRL` (0 rows) and `IQL_SAC_FineTuning_Walker2d` (653 rows), started 97, 96 and 115
days before this was written. `python openai_walker/mlflow_backend.py stale-runs --apply`
closed them and stamped a `closed_as_stale` tag on each, because `set_terminated` writes
today's date into `end_time` and a run that died in June would otherwise read as an 87-day
training session. Every trainer already wraps its work in `with mlflow.start_run(...)`, so
these are abandoned-by-kill runs, not missing `end_run` calls, and they are the reason an
artifact can be absent while its metrics are present.

## 🔬 Reproducing and measuring

```bash
python -m unittest discover -s tests -t .   # 41 tests; 7 skip on gymnasium<1.0 (phase-4 venv)
python bench_env.py --seconds 4             # env throughput, physics vs Python split
python bench_mjx.py --sizes 32,128          # MJX/JAX batched stepping
python verify.py                            # Phase-2 artifact check (exits 2 when missing)
python bench_device.py                        # SAC update cost, CPU vs CUDA
```

`verify.py` currently exits non-zero because `dataset.csv` is not in the repository — see the
Phase-2 caveat. That is the accurate answer, not a defect in the checker.
