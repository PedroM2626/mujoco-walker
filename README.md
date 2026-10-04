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

Phase 1's properly trained models were described here as "the SAC walkers in
`checkpoints/walker_target_v1` (40M steps)". **They do not walk to the target, and that is now
measured rather than inferred.** That directory holds 41 actor checkpoints of one real
40M-step run at `num_envs=32` (its own metadata says `env_version=
standup_balance_walk_curriculum_v4`, `target_forward_velocity=10.0`); 10 of them, spaced 1M to
40M, scored over 20 seeded target-phase episodes each
(`benchmarks/target_learning_curve.json`, `python eval_phase1.py --model ...`):

| SAC checkpoint | episodes inside the 0.45 m radius | median closest approach | median mean `x_velocity` |
|---:|---:|---:|---:|
| 1M | 0/20 | 3.03 m | +0.014 m/s |
| 3M | 1/20 | 3.11 m | +0.012 m/s |
| 10M | 1/20 | 3.05 m | -0.001 m/s |
| 20M | 0/20 | 2.98 m | +0.003 m/s |
| 40M | 0/20 | 3.50 m | +0.000 m/s |

The other five rows are in the JSON and say the same. Targets spawn 2-5 m away, so a median
closest approach of ~3 m with a median forward speed of ~0 m/s means the policy is not
approaching: **2 of 200 episodes** ended within the radius, and the best approach in any of them
was 0.31 m. Re-running the identical protocol against the v4 environment those checkpoints were
actually trained in (`python eval_phase1.py --env-commit 8d37846 ...`, written to
`benchmarks/target_learning_curve_envv4.json`, 80 episodes) gives 0/80
inside the radius and +0.006 m/s at 40M - so this is not an artefact of the env having changed
under them, and it is not "not enough steps": the curve is flat from 1M to 40M. `walker_recovery_v1`
was fine-tuned from this same base policy, which is the honest context for the Phase-3 result
that the recovery task vector dominates the merges.

The scorer used to be a different reward from the one the agent optimised. That is now fixed, and
fixing it changed the shape of the diagnosis, so both halves are recorded.

**The protocol defect and its fix.** `train_walker.py` shaped the reward with its own kwargs
(`standing_reward=0.0`, `target_direction_reward_weight=200`, `target_progress_reward_weight=300`,
`stability_reward_weight=20`, `stillness_penalty_weight=5`, `lateral_drift_penalty_weight=3`) while
the evaluators called `gym.make(...)` with the **environment defaults** (`standing_reward=50.0`,
direction 80, progress 200, stability 30, stillness 2, drift 2). Same trajectory, two quantities:
the defaults hand the 40M policy **+4.91/step of `standing`** that training paid zero for, and cut
its walking term from 5.44 to 2.18 per step. Measured in `benchmarks/reward_term_breakdown.json`
(the v8 recording) and reproduced by `python bench_reward_terms.py`.

The fix is one shared dict, `envs/reward_shaping.py::TRAINING_REWARD_KWARGS`, imported by the
trainer and by `eval_phase1.py` / `evaluate_merging.py`; new SAC/TD3/PPO checkpoints carry their
effective `reward_kwargs` (and their `target_forward_velocity`) so a scorer reads the reward off
the artifact instead of guessing, and `reward_kwargs_for()` falls back to the trainer shaping for
the historical SAC runs and to the environment defaults for ARS/REDQ/Dreamer, which never used
shaping at all. `--reward-weights env-default` reproduces the old protocol, and every scored row
prints and stores which reward it used.

Re-scoring the same 200 episodes with the reward these policies actually optimised
(`benchmarks/target_learning_curve_v9_trainreward.json`) moves the returns by an order of
magnitude and moves the walking conclusion not at all: **still 2 of 200 episodes inside the
radius** (the same two - 3M and 10M - and the same 0.31 m best approach), median closest
approach 2.72-3.50 m, median speed -0.005 to +0.047 m/s. Scoring those checkpoints in the v4 MDP
they were built for instead
(`benchmarks/target_learning_curve_envv4_trainreward.json`, 80 episodes) gives 0 of 80. Whatever
reward function and whatever environment revision you ask the question in, the answer is the same.

**Posture is now a bonus, not a punishment (env v9).** Below `z=0.65` the height term paid nothing
while `low_upright_penalty` charged `20 x (0.85 - z)` every step, so posture entered the return as
a punishment (-11.15/step on the 40M policy): a robot near the floor had no positive reward to
rise toward, only a smaller negative one.
v9 sets that penalty's default to 0 and grades the height bonus from the floor up
(`height x (1 + upright)/2`, 0 prone, 1.0 standing), which keeps the ordering and the get-up
signal while making the return answer "what did you achieve" instead of "how fallen are you". The
risk of removing a fall cost is that collapsing toward the target becomes the best-paying
behaviour, so `bench_reward_terms.py` now probes it explicitly, 10 seeded episodes per behaviour
(`benchmarks/reward_term_breakdown_v9.json`, per step, training weights):

| behaviour | v8 (punished posture) | v9 (posture as bonus) | mean `standing_gate` |
|:---|---:|---:|---:|
| SAC 40M policy | +8.21 | **+23.01** | 0.122 |
| commanding zero | -12.33 | +2.83 | 0.005 |
| every actuator at full extension | -23.60 | -5.32 | 0.004 |
| `abdomen_y` at +1 (topple probe) | not probed | +0.27 | 0.003 |
| `abdomen_y` at -1 (topple probe) | not probed | -4.93 | 0.001 |

No scripted collapse out-earns staying upright, so the exploit the change could have opened is
closed, and `tests/test_env.py::TestPostureIsABonus` pins it. The column that explains why 40M
steps did not buy walking is `standing_gate`: every locomotion term in the target phase is
multiplied by it (`walker_ragdoll_env.py`: `weight x standing_gate x min(v_toward, tvf)`), and the
policy averages **0.122**. It spends ~88% of each episode outside the window in which moving toward
the target pays anything at all, which is a much stronger statement than "3% of the walking signal"
- the walking reward is not merely small, it is mostly unreachable from the posture the policy
settled into.

v9 is a new MDP, so `ENV_VERSION` moved and the trainer's existing guard refuses old checkpoints
unless `--allow-mismatched-env-version` says otherwise; `eval_phase1.py --env-commit <rev>` scores
them in their native revision instead. The reward goldens in `benchmarks/walker_ragdoll_golden.json`
were regenerated for v9 (`obs_sum` is bit-identical to the v8 recording - the observation math did
not move; `reward_sum` did, e.g. target 300-step 598.46 to 5576.32).

The first Phase-1 run trained on v9 is ARS at its full 1M budget
(`benchmarks/phase1_ars_v2_1m_v9.json`, 20 seeded target-phase episodes, scored with the
environment defaults because `train_ars.py` never used the trainer shaping): mean 8679.40,
median 8898.85, std 4904.52, 0.35 falls per episode, **0 of 20 inside the radius**, mean closest
approach 3.015 m, mean x-velocity -0.0216 m/s. Same verdict as v8's ARS at the same budget, which
is the comparison that makes the v9 change safe to have made: the reward shape moved the returns,
not the behaviour.

What the task geometrically requires is not in dispute: `timestep=0.002` with `frame_skip=5` makes one env step 0.01 s, episodes are capped at 1,000 steps (10 s of simulated time), targets
spawn 2-5 m away and the success radius is 0.45 m. Reaching the near target needs 0.2 m/s
sustained, the far one 0.5 m/s; at the env's own nominal 0.8 m/s the walk itself is 250-625 env
steps of a 1000-step episode. Every checkpoint above averages ~0.0 m/s, so none of them is
anywhere near that floor. See "What a training run costs" in the throughput section for what a
retraining would take in wall clock.


The scale of that table matters: REDQ and Dreamer wrap the environment in `NormalizeReward`,
so the `Mean return` column is a normalised sum, not the reward the raw environment reports.
On the raw scale the same checkpoints are indistinguishable from an inert robot. Measured in env
v8 over one seeded 300-step rollout each (`benchmarks/phase1_inert_reference_v8.json`,
`python bench_inert_reference.py --tag v8 --env-commit 2d59b7b`): commanding zero **-3775.76**,
uniform random -4146.46, REDQ -4248.46, Dreamer -3588.23, ARS -4105.24 - and **none of the five
moves more than 0.4 m**.

Two things are worth saying about that sentence. It used to read -3826.77 / -3836.79 / -3887.55 /
-3857.06 for those same rollouts: those figures were transcribed by hand into a note string and
**could not be reproduced from any recorded protocol**, so they are replaced here and in
`benchmarks/phase1_evidence.json` by the artifact above. Getting them reproducible also exposed a
second defect: the Dreamer actor samples its stochastic state, so scoring the same checkpoint twice
gave a different number three times (-3425.86, -3637.40, -3278.44) until the policy RNG was seeded -
`eval_phase1.py` now seeds per model, the same fix `openai_walker/evaluate_all.py` needed.

And in env v9, where posture is a bonus instead of a punishment, all five land in one band -
commanding zero +602.63, random +237.06, REDQ +781.97, ARS +729.58, Dreamer +1691.21 - with the
*same* sub-0.4 m displacements and a `standing_gate` of 0.000 for every one of them
(`benchmarks/phase1_inert_reference_v9.json`). The return scale moved by +4.4k to +5.3k; the
behaviour did not move at all. That is why a return is a bad thing to pin "it does not walk" on, and
why `tests/test_phase1_behaviour.py` now pins the claim three ways: the loading contract of each
evidence checkpoint (recorded observation width, matching `obs_rms`, actions inside the action box,
`deterministic=True` really being deterministic), the "not better than doing nothing" return margin,
and a new **displacement** assertion - under 0.5 m of travel in 300 steps for both the REDQ evidence
actor and commanding zero - because displacement means the same thing in every reward revision while
a return does not.

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
1. First, we pre-trained a base agent intended to walk (`walker_target_v1` - 40M steps).
   *"Intended"*, not "walking perfectly": measured over 200 seeded target-phase episodes across
   its 1M-40M checkpoints it enters the 0.45 m success radius twice, with a median forward speed
   of ~0.00 m/s (see the Phase-1 section). Everything below still holds as stated - the merging
   experiments compare two policies that share an initialisation - but the "walking policy" leg
   of the story is a standing/half-fallen policy, which is what the Phase-3 numbers show when
   they rank the recovery vector above the walking one.
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

⚠️ **That table is the retired protocol** (env v8, scored with the environment's default reward
weights). Re-run identically - `evaluate_merging.py --num-episodes 100 --seed 11`, 400 episodes,
same checkpoints and same seeded resets - but scored with the reward the experts actually
optimised, in env v9 (`benchmarks/phase3_merging_100ep_v9_trainreward.json`):

| Merging Strategy | mean, retired protocol | mean, corrected | median, retired | median, corrected | Falls / episode |
|:---|---:|---:|---:|---:|---:|
| Hardcoded Supervisor | 13549.68 | **27852.23** | 831.31 | **16525.67** | 1.31 |
| Mixture of Experts (MoE) | 9199.76 | **24062.63** | -4391.54 | **11046.13** | 0.92 |
| Task Arithmetic | -17410.69 | **1004.07** | -15126.37 | **1766.85** | 0.24 |
| Weight Averaging (50/50) | -18313.84 | **-1574.96** | -13297.76 | **1476.67** | 0.17 |

The ordering is identical, so the Phase-3 conclusion - a routed or supervised policy beats both
static merges - does not depend on the protocol. What the protocol was doing is hiding the sign:
under the retired scoring Task Arithmetic and Weight Averaging looked catastrophic (-17.4k and
-18.3k with -15.1k/-13.3k medians), and almost all of that depth was the posture punishment the
scorer was applying - `-low_upright x 20 x (0.85-z)` on robots that spend most of an episode on
the floor. Scored with the reward they were trained against, both merges sit near zero with
positive medians (1004.07 / -1574.96, medians 1766.85 / 1476.67), which is a much more defensible
statement of "the merge degrades performance" than "the merge is catastrophic".

Points 2 and 3 below are about the `obs_rms` input bug and were measured under the retired reward
protocol; they have not been re-run under v9, and their conclusion (the input error, not the
router) is a statement about inputs, so it is independent of the scoring. Points 1, 4 and 5 are
quoted from the corrected artifact where it matters: falls per episode (1.31 supervisor, 0.92 MoE,
0.24/0.17 merged) and the upright-at-end share (1% / 2% / 0% / 0%) are identical under both
protocols, because posture at the end of an episode is not a reward quantity. What changed is the
distribution reading in point 1.

Read this table before citing it:

1. **How much the mean is carried by outliers depends entirely on which reward you score with,
   so both are recorded** (`mean_without_best`, `std_over_abs_mean` and the tail shares are fields
   of both JSON files, recomputed from the logs by `python summarize_benchmarks.py`):

   | strategy | protocol | mean | median | std/\|mean\| | drop best 1 | drop best 5 | share below -10k |
   |:---|:---|---:|---:|---:|---:|---:|---:|
   | Hardcoded Supervisor | retired | 13549.68 | 831.31 | 2.31 | 12239.87 | 8889.50 | 28% |
   | Hardcoded Supervisor | corrected | 27852.23 | 16525.67 | **1.06** | 26708.73 | 23429.37 | **1%** |
   | Mixture of Experts | retired | 9199.76 | -4391.54 | 3.52 | 8042.06 | 4437.68 | 33% |
   | Mixture of Experts | corrected | 24062.63 | 11046.13 | **1.27** | 23078.68 | 19515.37 | **0%** |
   | Task Arithmetic | corrected | 1004.07 | 1766.85 | 8.12 | 754.37 | 146.13 | 8% |
   | Weight Averaging | corrected | -1574.96 | 1476.67 | 6.04 | -1825.14 | -2621.31 | 24% |

   The retired protocol's famous bimodality was mostly the scorer. Under the environment defaults,
   28% of supervisor episodes and 33% of MoE episodes scored below -10k, and dropping the single
   best episode moved the supervisor mean by 1310 points - that is the signature of a return that
   is negative whenever the robot is on the floor. Scored with the reward the experts optimise, the
   supervisor's std/|mean| falls from 2.31 to 1.06, its below--10k share from 28% to 1%, and its
   median rises from 831 to 16526: the distribution is no longer two humps, it is a wide single one
   with 44% of episodes above +20k. What *is* still outlier-driven is the merged policies - under
   the corrected scoring their means are near zero with std/|mean| of 8.12 and 6.04, which is why
   their median (1766.85 / 1476.67) is the number to quote, not the mean. Rank by median, or say
   which one you mean.
2. **The previously published ordering was one specific line of code, and it has now been
   reproduced rather than asserted.** The retired table (`git show 32da152:README.md`) read
   -11191.88 / -11692.61 / -11763.92 / -16958.91 for supervisor / weight averaging / MoE /
   task arithmetic, which ranked Weight Averaging second ("highly robust, counter to
   intuition"). The cause was in `evaluate_merging.py`: it called
   `single_agent.get_action(obs_49)` on raw observations and never applied the `obs_rms`
   pickled inside each checkpoint. Re-running the *corrected* script with `--raw-obs` - which
   now exists solely to reproduce that input path - at the same 100 episodes and seed 11:

   | Merging strategy | corrected harness, retired reward (`obs_rms`) | raw obs = old harness | retired publication |
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

Headline numbers are the **50 seeded episodes** below, re-measured on 2026-10-01; the
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
paired-t and Wilcoxon p-values, and how many of the 50 episodes each side wins. Four distinct
pairs are tested: two are ties, one (Extra Trees against BCQ) is won on rank but not on mean,
and one - BCQ against the teacher - is a clear gap, in the direction the retired leaderboard
had backwards.

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
| **MaxEnt IRL** | 279.33 | 34.47 | 197.38 | 354.70 | Linear reward model ($r = \theta^{\top}\phi$) too weak for bipedal locomotion. |
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
7. **Every retired single-episode draw is now placed inside the distribution the same weights
   produce today**, so "the n=1 table was a poor estimator" can be said per model instead of as
   one sentence for thirteen numbers (`benchmarks/phase4_n1_vs_50ep.json`, regenerated by
   `python summarize_benchmarks.py`). None of them is a measurement error: all thirteen land
   between the min and the max their own 50 seeded resets span. What varies is how far from
   their own mean each one sat. BC's 1728.81 is **-2.77 sigma** and 48 of 50 episodes beat it;
   BCQ's 3897.63 is **+1.09 sigma**, reached or exceeded in 10 of 50, with a max of 4049.40 - a
   legitimate high day, not a defect; the teacher's 3865.41 is +0.48 sigma; and the two rows that
   the old table over-credited relative to themselves are BC+SAC (Constrained) at +1.71 sigma
   (only 2 of 50 episodes do as well) and BC+SAC (Regularized) at +0.75 sigma. Read together, that is
   what the retired leaderboard was: not wrong numbers, but thirteen draws ordered by luck, where
   the model with the widest spread (BCQ, sigma 1066) had the most room to look like a winner.
8. **The claim the retired leaderboard actually made is now tested, and it fails.** Its top row
   asserted BCQ "beat the online teacher". Paired over the shared resets, BCQ is **784.34 below**
   the teacher, 95% interval [-1122.42, -439.76], paired-t p=0.0001, Wilcoxon p<0.0001,
   dz=-0.622 - it wins **13 of the 50** episodes. This is the only significant difference among
   the pairs tested, and the sign is the opposite of what was published: the gap is not too small
   to measure, it is large and points the other way.

9. **A second protocol, recorded back in June, reproduces the level.** The teacher run's own
   TensorBoard log holds 50 evaluations of 100 deterministic unseeded episodes each (SB3's
   `EvalCallback` default) - a different harness, a different episode count, and the only
   evaluation series in the repository that predates the retracted README. Its last 11 checkpoints
   after 400k steps give min **2389.32**, max **3891.31**, mean **3534.66**, against the seeded
   50-episode table's **3516.95**: 0.5% apart. Its final evaluation, **3857.97**, is within 7.44 of
   the single-episode record's 3865.41. So the ~3.5k level is not an artefact of one harness, and
   the two ~3.86k readings are both single evaluations - the same selection effect that produced the
   retired table. The extraction is `benchmarks/teacher_eval_curve.json`, and it settles a negative
   too: those are the teacher's numbers, and nothing in the repository logged a score for any of the
   13 offline policies before 2026-06-26.

### Historical record: one unseeded episode per model (`final_results.txt`)

Kept for the numbers themselves - this is the file the earlier version of this README
contradicted - and for the handful of facts only this record carries, listed under the table.
The scores are single unseeded draws: compare them with the 50-episode table above rather than
quoting them, and read `benchmarks/phase4_n1_vs_50ep.json` for where each draw sits inside the
distribution its own weights produce today.

**Which of these can still be re-measured.** `.gitignore` excludes `*.pt`, so none of the PyTorch
Phase-4 policies is version-controlled - the teacher is the exception, committed as
`sac_walker2d_final.zip` - and for the rest the files on disk are the only weights. Seven of the
thirteen scored here are also archived as MLflow run artifacts (`bc`, `iql`, `cql`, the three
`bc_sac` variants, `cql_sac`) and those archived bytes are identical to the ones
`evaluate_all.py` scores today, so their retired numbers can be replayed against the same policy.
The other six are not archived, and their provenance is the file mtimes: `bcq_*` were written
2026-06-26, the morning this capture was taken, so BCQ's 3897.63 is a measurement of the policy
still on disk; `gail_model.pt` dates to 2026-06-10 19:03 and `airl_model.pt` to 2026-06-13.

That distinction is what makes one retired figure stand out. GAIL is effectively deterministic
here - 50 different seeded resets span 997.31 to 998.74, a band 1.43 points wide - and both
evaluation paths run it through the same action head (`GAILActor` is defined in `play_race.py`,
which `evaluate_all.py` imports it from; the latter's `is_gail` branch is a no-op that repeats
the default). So episode luck cannot move GAIL by more than about a point and a
half. The earlier version of this README quoted 1016.41 for it: **17.67 above the best of the 50
seeded episodes**, which is outside anything this protocol produces. The checkpoint on disk
predates the commit that quoted that figure by twenty minutes, so if it was rewritten inside that
window nothing records what it held. It is listed among the unsupported figures above, and this
is the one of them whose gap has no candidate cause on record - BC's retired 3837.80, by
contrast, is +0.47 sigma inside its own distribution and needed no explanation at all.

| Model Architecture | Final Score |
|:---|---:|
| **Batch-Constrained Q-learning (BCQ)** | **3897.63** |
| **Teacher (Online SAC)** | 3865.41 |
| **BC+SAC (Regularized)** | 3396.81 |
| **Behavioral Cloning (BC)** | 1728.81 |
| **BC+SAC (Naive)** | 1369.87 |
| **Decision Transformer (DT)** | 1288.02 |
| **Inverse RL (GAIL)** | 997.57 |
| **CQL+SAC** | 401.80 |
| **BC+SAC (Constrained)** | 338.67 |
| **CQL Offline** | 315.37 |
| **MaxEnt IRL** | 269.61 |
| **Inverse RL (AIRL)** | -6.36 |
| **IQL Offline** | -15.81 |

**What only this record tells you.** Five facts here are about how the models were built, which no
score table can carry:

- **DT scored 1288.02 after 10 epochs (100k steps)** - the least-trained model in the race, and
  separately the widest spread in the seeded table (std 1101.18). Side by side, not cause and
  effect: nothing here separates under-training from the variance a Return-To-Go policy has anyway.
- **GAIL was trained to 1,000,000 steps and plateaued near 1000** - the dataset was too
  deterministic, the discriminator became a perfect judge, and the actor starved. That plateau, not
  a bad evaluation, is what its near-zero variance measures.
- **AIRL was rebuilt once**: the first version produced `NaN` gradients where deterministic dataset
  actions hit the `atanh` limits, and came back with spectral normalisation, action clipping
  ($\pm 0.95$) and an $h(s)$ shaping baseline. It still sits at -6, the measured value, where the
  old prose said "near -5". A 20-hour server restart killed a long run mid-training.
- **BC+SAC (Regularized)** means what it says: BC weights, then SAC exploration held near the clone
  by a BC loss. This record is where "below BCQ and the teacher, so not the champion an earlier
  version claimed" comes from; the seeded table says it also finishes below plain BC.
- **Extra Trees was never in this race**, so it was never ranked against the rest - and the
  "~2522 over 5 episodes / ~3900 in-distribution / ~900 extrapolating" breakdown once quoted here
  exists nowhere in the repo as a measurement.

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
python bench_env.py --reps 3 --seconds 4            # median of 3 reps, with the spread printed
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

⚠️ **These are steady-state rates: they time the stepping loop, not a training run.** The same
three configurations measured end to end (whole SAC runs, worker startup included) come out at
1.22x and 1.86x, not 1.5x and 6.5x — see "The whole stack, end to end" below.

⚠️ **How to read these numbers.** This is a laptop CPU whose clocks vary with power and
thermal state, and repeat runs of the identical command have ranged ~2x apart (the sync
n=32 configuration measured 1,408, 1,536 and 2,720 env-steps/s in three runs the same
minute-scale window apart). The absolute column is therefore indicative; the **ratios** are
what to trust, because each was measured back-to-back in one process against the same
baseline. Re-measure on your own hardware with `python bench_env.py --seconds 4` before
quoting a multiplier.

What the rewrite actually bought is a constant number of microseconds, not a percentage: the
Python around MuJoCo in one `env.step` fell from ~46% of wall clock to **7%** (23 µs of a
312 µs step, physics 289 µs), which is why the remaining headroom is in parallelism and in
the integrator rather than in Python.

**Re-measured in one process, three reps each** (`python bench_env.py --mode all --n 32
--seconds 4 --reps 3 --json benchmarks/throughput_replication.json`): the Python side came out at
**23 µs** again, exactly as published, while the physics side took 665 µs instead of 289 - so
the same run reports the Python share as 3% rather than 7%. That is the whole lesson about this
section's percentages: the µs are the measurement and the ratio is whatever the window's
clocks make of it. The spread between the three reps of the same configuration reached **77%**
on the physics row and 50% on the sparse-parallel row, so a single `bench_env.py` number on this
machine is not quotable at all, and the same-process sync-to-parallel ratio came out 8.0x against
the table's 6.5x (which was measured against the *pre-rewrite* baseline, so the two are different
comparisons, not a disagreement).

The tool also had a unit bug, found by this replication: `bench_vec_env` and
`bench_parallel_vec` timed one *vector* step and published the rate as if it were env-steps, so
the printed number for n=32 was 32x smaller than the table it was meant to be compared with -
and the README's own instruction here is "re-measure with `bench_env.py` before quoting a
multiplier". `steps_per_s` now multiplies by n, with `vec_steps_per_s` and the per-vector-step
`us_per_step` kept beside it.

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
   episode, so it is dropped from every other reply. Measured as an A/B it is the biggest single
   switch in the env path - 40% in the window that shipped it, 2.14x in the replication above -
   but the arms' own spread reached 50%, so treat it as "large, and not pinned".
   `--vec-dense-info` restores the dict.

**Physics was left alone on purpose.** `walker_ragdoll.xml` still compiles RK4 at
`timestep=0.002` with `frame_skip=5`. Euler measures 2.1x on physics-only, but diverges
from RK4 by `|dq| = 1.96` over 2,000 identical actions — enough that every checkpoint in
`checkpoints/` would be navigating a different MDP. The training budget (env steps) is
unchanged everywhere above; only wall clock moves.

### ⏱️ The whole stack, end to end: 1.6x at small budgets, 1.9x at collection-dominated ones

Multiplying the factors above is not a measurement and the product would be wrong: the device
factor already contains collection, while `bench_env.py`'s steady-state numbers exclude the
cost of *starting* the worker processes. `bench_stacked.py` runs whole trainers instead, arming
each change on top of the stack that was committed at 5920805 (that revision's env and trainer
are extracted with `git show` and aliased in as `envs.walker_ragdoll_env`, one directory deep so
the env still finds its own `walker_ragdoll.xml`). Best of 2 alternating reps, on a laptop that
also had a Dreamer run on it for the whole window:

| 20k-step SAC budget, 8 envs | wall clock | vs where we started |
|:---|---:|---:|
| original env + original trainer + sync + CPU | 131.8 s | — |
| + today's trainer (the correctness fixes alone) | 136.7 s | 0.96x |
| + rewritten env | 141.6 s | 0.93x |
| + parallel vector backend | 182.6 s | **0.72x** |
| + GPU learner | 140.2 s | 0.94x |
| rewritten env + **sync** + GPU learner = today's default at 8 envs | **84.4 s** | **1.56x** |

Two things fall out of that table that the steady-state benchmarks cannot show. First, at 8
envs the run is learner-bound, so the environment rewrite is *invisible* end to end (0.93x,
inside the 11% rep spread). Second, the parallel backend is a 28% **loss** at this size,
because 8 worker processes have to boot before the first step — the measured reason
`--vec-backend auto` refuses parallel below 16 envs.

Where the environment work does pay is the collection-dominated regime. Same harness, 32 envs,
600,000 steps with the update gate closed:

| 600k steps, 32 envs, collection only | wall clock | env-steps/s | vs committed |
|:---|---:|---:|---:|
| original env + sync | 540.9 s | 1,109 | — |
| rewritten env + sync | 444.8 s | 1,349 | **1.22x** |
| rewritten env + parallel | 290.9 s | 2,062 | **1.86x** |

`bench_env.py` reports 7,328 env-steps/s steady state for that last line; over 600k steps the
same configuration delivers 2,062, because ~200 s of the 290.9 s is 32 child processes importing
torch and mujoco. With the measured startup and the measured 6,694 env-steps/s wrapped-stack
rate, a 1M-step collection lands at ~2.1x over the committed sync backend.

**What the repo's own 40M-step run cost.** `checkpoints/walker_target_v1/` holds 41 actor
checkpoints of one real run at `num_envs=32`; their mtimes date the run itself: 37.6M env steps
in 547 min of continuous training = **14.5 min per 1M steps (1,146 env-steps/s; segments range
10.5-16.7 min)**, spread over 23.5 h of calendar time because of two pauses. That is the same
shape of run as the 1,109 env-steps/s line above, which is the cross-check that makes the
summary of this section credible: **a committed training run gets ~1.6x faster at 8-env budgets
and ~1.9x when collection dominates — not 6.5x.** The time that is left is inside the update
loops, which is the next section.

Evidence: `benchmarks/throughput_stacked_ladder.json` (raw per-rep seconds, the steady-state
contrast, and the checkpoint-mtime arithmetic).

**What a training run costs.** The two ladders above are chosen regimes, so the last one replays
the *actual* configuration of the 40M-step run - `num_envs=32`, `learning_starts=10000`,
`task_phase=target` - for 400,000 steps on both stacks, back to back in the same window
(`benchmarks/throughput_stacked_ladder_n32real.json`, `python bench_stacked.py --ladder n32_real`):

| 400k steps, 32 envs, target phase | wall clock | min per 1M env steps | 40M extrapolated |
|:---|---:|---:|---:|
| committed stack (5920805 env + trainer, sync, CPU) | 927.4 s | 38.6 | 25.7 h |
| today's default (rewritten env, parallel, CUDA) | 821.1 s | 34.2 | 22.8 h |

**1.13x.** At 32 envs a vector step carries 32 environment steps but only one gradient update,
and the parent process now spends its time shuttling 32 observations and 32 actions through
Windows pipes - so neither the physics rewrite nor the GPU learner is the bottleneck any more,
and the worker startup (~200 s) is paid in full by a run this length.

That is also the honest answer to "how much did training time improve", for as long as the update
loops are left alone: **1.5-1.6x where the learner dominates at small `num_envs`, 1.9x where
collection dominates, 1.13x at the exact configuration this repo's flagship run used.** Multiply
nothing across rows. And note the third
column of the table above disagrees with history: the checkpoint mtimes say the same committed
code ran at 14.5 min per 1M steps in May, against 38.6 min per 1M for that identical code replayed
here on an AC-powered, otherwise idle laptop - a 2.7x machine-state gap that no code change
explains (`OMP_NUM_THREADS=1` did not move it: 24.2 ms per CPU update against 27.5 ms at 24
threads). Absolute minutes-per-million on this box are therefore per-window, not per-commit; the
back-to-back ratios are the ones to reuse.

Where the numbers above stop is exactly where the environment work stops mattering. SAC spends its
time in a few cheap gradient steps per 32 environment steps, so no physics change reaches it; REDQ
spends ~2% of its wall clock in MuJoCo and the rest in per-critic kernel launches, so the lever
there was batching the ensemble - **2.15x end to end, 3.4x on the update phase**, in the
learner-side section below. "Why is training still slow" and "the environment is already fast" are
both true, and the second one is why the first one cannot be fixed from the physics side.

### 🧮 The learner side: what was measured, what shipped, what was rejected

Collection is not the bottleneck for the Phase-1 algorithms, so "faster training" has to be
answered per trainer, at the same env-step budget. Every number below is wall clock on this
laptop (RTX 4070 Laptop 8 GB, 32 threads, `.venv`), one process at a time:

| Trainer | Config | Measured | 1M env steps |
|:---|:---|:---|:---|
| `train_ars.py` | linear policy, 10 directions | 1,005,153 steps in 544 s = 1848/s | 9 min |
| `train_dreamer.py` | 4 envs, update each collect step | 5,000 steps in 37 s with the update gate closed; 14,000 in 898 s in one A/B window and in 205 s later, same build | ~4-18 h |
| `train_dreamer.py`, captured update | same config, and the CUDA default now (`--no-update-graph` opts out) | 100 updates in 4.1 s against 17.6 s eager, in the same window | its own projection says 3.6 h; measured later at two budgets: **1.18 h** |
| `train_redq.py` | 16 envs, `utd_ratio=20`, ensemble 10 | 10,000 steps in 478 s | ~13 h |
| `train_redq.py`, batched ensemble | same config, `--ensemble-impl batched` (default now) | see the A/B below | ~4 h in that window |

The Dreamer row was the whole story of the eager build: with its update gate closed this trainer
collects 5,000 steps in 37 s, which is where "~96% of its wall clock is the learner step, not the
physics" came from. That ratio was measured against an eager update, in a window that had other work
in it. Measured again after capture, in a clean window and with two independent instruments, one
iteration of the shipped loop is 16.96 ms, of which the captured update is 12.75 ms (**75.2%**) and
the whole collection side is 4.21 ms. The environment is still not what costs here - and the headroom
on it is now about a quarter of the loop, not 96% of it.

The 4x spread on that row is not measurement noise and it is not the code: this is a laptop
whose GPU is also serving Brave, Medal.tv and Overwolf (`nvidia-smi` shows five desktop
compute contexts holding ~2.1 GB before any training starts). Two runs of the same build
measured 15.6 and 68 env-steps/s. Treat every ETA in this table as a range, and quote a
rate only with the wall-clock window it came from.

**Shipped: the world-model KL, batched over time.** It built 49 pairs of
`torch.distributions.Normal` and called `kl_divergence` once per timestep - 64.9 ms of a
207.6 ms world-model update, as much as the entire RSSM forward pass, for a 256-unit hidden
state and a 16x50 batch. Stacking the lists into one `(T, B, z)` pair measures 0.78 ms for the
same value (0.492967 against 0.492967; 6e-8 relative, float reassociation only). Isolated:
**207.58 ms -> 100.54 ms, 2.06x**. End to end, A/B on the real trainer with everything else
equal (same seed, same args, same idle machine, 14,000 steps): **989 s -> 898 s = 1.10x**. The
2.06x is the truth about the KL and the wrong number to quote for the run, because the
imagination rollout and the actor-critic losses are Python loops too - that is where the same
technique would go next. Fixing it also exposed a wrong pairing: `prior_means[t]` predicts
step t+1 while `post_means[t]` is step t's posterior (and `post_means[0]` is the pre-action
posterior at step 0), so the regulariser was pulling each step's prior toward the previous
step's encoder; the value moves 0.4930 -> 0.4947 at initialisation.

**Shipped: the REDQ critic ensemble as one batched pass.** This is the biggest learner-side win
in the repository, and it is the answer to "the environment is fast, why is training slow". A REDQ
gradient step touches all N=10 critics for the target, all N again for the critic loss, all N a
third time for the actor, and then soft-updates the target copy with 3N tiny kernels - on every
gradient step, not on a frequency. On 256-unit nets that is kernel-launch time, not arithmetic, so
`--utd-ratio 20` runs at ~32 env-steps/s while one environment steps physics at 1,560/s: the
MuJoCo inside a REDQ run is about **2% of the wall clock**, and no amount of physics work can make
that run faster. `BatchedSoftQEnsemble` (`train_walker.py`) holds the same weights as stacked
(N, in, out) tensors and evaluates them in one `baddbmm`. Isolated critic step at batch 256,
N=10: **8.61 ms loop against 0.95 ms batched on the GPU** (26.6 ms against 7.85 ms on CPU), and the
3N-kernel target update goes 3.38 ms -> 0.15 ms. The arithmetic is equal to 3.73e-7 relative on the
forward and 5.52e-7 across all 30 weight gradients, and the initialisation consumes the same RNG
stream so a seeded run starts bit-identically (`tests/test_redq_ensemble.py`). Those milliseconds
come from `python bench_redq_ensemble.py --isolated-only` on a window that also carried the live 1M
run, so the 9.06x and the 22.53x are the parts worth reusing, not the absolute times.

Inside the real trainer (`python bench_redq_ensemble.py`: UTD 20, N=10, 10k env steps, updates
from 2k, two reps, and a collection floor per implementation measured with the update gate
closed):

| REDQ at n=16 parallel | total s (rep 1 / rep 2) | gradient phase s | env-steps/s |
|:---|---:|---:|---:|
| loop ensemble | 309.3 / 434.1 | 220.9 | 32.3 |
| batched ensemble | 143.7 / 163.9 | **64.9** | **69.6** |
| collection floors, loop / batched | 88.4 / 78.8 | - | 113.2 / 126.8 |

**3.4x on the update phase and 2.15x end to end** at the shipped configuration. The subtraction is
licensed by the two floors agreeing to 12% - same physics, same vector backend, no gradient work.
Divide the 1M-step budget by those two rates and it is 8.6 h becoming 4.0 h, and those hours belong
to this window the way every other absolute figure here does. The live 1M run that shared the box
with this A/B is still on the loop path - it started before the batching existed, and `--resume`
will not move it across - so it says nothing about the ratio; what does is two arms, same seed, same
args, one running at a time. `--ensemble-impl loop` keeps the original path for exactly this A/B.

The same A/B at n=4 on the sync backend, where neither worker startup nor IPC sits inside the timed
phase, is the unobstructed version of the same comparison: 4k env steps, updates from 400, 898.6 s
on the loop path against 224.8 s batched - **4.0x end to end, 4.4x on the update phase**. Its two
collection floors (12.2 s loop, 23.2 s batched) disagree by 90%, which is why that arm is quoted
from the whole-run times and the subtraction is treated as indicative rather than licensed; the
n=16 floors above, which agree to 12%, carry the load.

Two limits, said plainly. Checkpoints are written in the old `nn.ModuleList` key layout from both
paths, so `evaluate_merging.py`, the eval scorer and the merge arithmetic are untouched - but
Adam's state is per parameter and the two layouts hold 6 versus 3N tensors, so `--resume` now
*refuses* a checkpoint written by the other implementation instead of quietly rebuilding the
optimizer (`--init-from-run-id` remains the weights-only route across them). And the same
treatment is owed to Dreamer too, but not in that form: its per-timestep work is a recurrence, so
there is nothing to stack - what shipped instead is the graph capture below.

**Shipped: the Dreamer update as one graph replay.** The ensemble trick does not transfer here,
because a Dreamer update walks 49 timesteps of world model plus `--imag-horizon` more of imagination
and `h_t` depends on `h_{t-1}` - there is nothing to stack. What is left is dispatch. Running the
real trainer under `torch.profiler` counts **14,400 aten calls per update** (1,439,168 over 100
updates: 43,245 `aten::linear`, 249,460 `aten::t`, 7,814 `rssm_transition` calls) against six
256-unit MLPs, and self CPU time lands at 1.04x self CUDA time - the GPU is waiting for Python.
`CapturedDreamerUpdate` (`train_dreamer.py --update-graph`) records that update once and replays it
as one launch.

| Dreamer, 5,400 steps at n=4 (100 updates) | total s (rep 1 / rep 2) | ms per update | projected 1M |
|:---|---:|---:|---:|
| eager update | 33.3 / 39.7 | 176.2 / 248.0 | 13.0 / 17.9 h |
| captured update | 19.2 / 20.5 | **40.9 / 39.1** | **3.6 / 3.5 h** |
| collection floors, eager / captured | 15.7 / 15.1 and 14.9 / 16.6 | - | - |

**4.31x and 6.34x on the update phase, 1.74x and 1.94x end to end at this budget.** The projection
column is the one that matters for a real run, because 100 updates is not what a 1M-step budget does:
at 1M the same rates buy **3.57x and 5.16x** (248,750 updates plus the floor arm's own collection
cost - `benchmarks/dreamer_update_graph_ab.json` stores that as derived arithmetic, not as a run that
was performed, and the two reps disagree with each other across that range for exactly the reason the
machine-state caveat above gives). That column no longer stands as an estimate of a real run: two of
its terms are fixed costs divided by a step count, and "Cross-checked against the trainer itself"
below carries the measurement that replaces it. The other thing worth reading off the table is the
spread: the
eager arm's per-update cost moved 40.7% between reps on a box that also had the live REDQ run on it,
the captured arm's moved 4.6%. Capture does not make the kernels faster; it deletes the host from the
loop.

**What is left after capture, priced instead of assumed.** This section used to point at the
imagination rollout and the actor-critic losses as "where the same technique would go next".
`python bench_dreamer_update.py --mode scaling` measures that claim on isolated `dreamer_update`
calls at the shipped sizes (seq 50, batch 16, horizon 15): on the captured path one imagination step
costs **0.218 ms** and one world-model timestep **0.138 ms**, so of a **12.62 ms** update the
imagination loop is about 3.3 ms and the world model about 6.9 ms. Only the captured arm is quoted,
because only it repeats: across three windows this session the captured update came out 12.41,
12.62 and 12.62 ms while the eager arm - identical code, same script - gave 132.63, 254.94 and
293.56 ms - the first of those is still in the artifact's git history. An eager timing on this
box measures what else was running, so the JSON keeps it and the prose does not argue from it.
Neither loop can be collapsed the way the REDQ ensemble was: `h_t` depends on `h_{t-1}`, and prior and posterior take different inputs. The only
fusion left is across heads that share an input - `reward_net` and `continue_net` have identical
trunk shapes and one optimizer - about a fifth of the imagination loop's launches, ~6% of the update
on paper. That one was built, measured and shipped: see "Shipped: the two imagination heads as one
pass" below, where the prediction meets a box whose timing noise is the same size as the effect.

**Where a whole iteration goes, now that the update is cheap.** The paragraph above prices the inside
of an update; this prices the loop around it, because the claim this section inherited - "~96% of
Dreamer's wall clock is the learner step" - was measured when the update ran eager at 293 ms.
`python bench_dreamer_update.py --mode loop-split` rebuilds one iteration from the trainer's own
pieces (`build_vec_env` with the trainer's flag set, the same wrappers, `WorldModel`/`DreamerActor`,
`SequenceReplayBuffer`, the extracted `advance_recurrent_state`, `CapturedDreamerUpdate`), times every
segment with a `synchronize()` at each boundary, and repeats 25 iterations in one process and one
window (`benchmarks/dreamer_loop_split.json`, 2 other CUDA contexts):

| segment | median ms | of the iteration |
|:---|---:|---:|
| policy pass (`actor.get_action(...).cpu()`) | 0.53 | 3.4% |
| `envs.step` (4 envs, sync) | 1.62 | 10.4% |
| recurrent-state pass (encoder + RSSM + posterior + reset loop) | 0.82 | 5.3% |
| replay buffer write | 0.02 | 0.1% |
| replay sample + starting latents | 0.34 | 2.2% |
| captured update | 12.27 | **78.6%** |
| attributed iteration | 15.61 | 100% |

The same iteration with no boundary syncs is **15.70 ms**, which is 254.9 env-steps/s at the shipped
`--num-envs 4`. Two things follow. The learner still dominates - so "the environment is already fast
enough" remains true - but the collection side is now only **21.4%** of the loop, so no work on the
env path, the wrappers or the state pass can buy more than a fifth, and the sentence above about the
imagination loop is where the remaining ceiling actually is. And the recurrent-state pass is 0.82 ms
of that 3.34 ms, which is why the collection loop was left alone rather than restructured: it is not
where the time is.

**Cross-checked against the trainer itself, and the 1M projection replaced.** The split above rebuilds
the loop; `python bench_dreamer_update.py --mode real-rate` runs the shipped command instead, at two
budgets (20,000 and 40,000 env steps), every cell twice, with the four cells rotated - because the
first process of a batch pays a cold start, and an early unrotated version of this measurement
reported a *negative* slope. Subtracting the two budgets cancels the per-process fixed cost, and the
arm that never updates (`--learning-starts` above the budget) measures the collection side alone:
**16.96 ms** per iteration with the update against **4.21 ms** without it, so **12.75 ms** is the
captured update - next to the 12.27 ms the in-process split attributed and the 12.62 ms the isolated
sweep recorded, three instruments agreeing within 4%. At **235.9 env-steps/s** that is
**1.18 h per 1M env steps**, where this section's own projection said 3.5-3.6 h.

The projection was arithmetic on the 5,400-step A/B, and both of its terms were contaminated by the
fixed costs it divided: 100 updates is so few that the one-time graph capture sits inside the
per-update figure (40.9 ms measured there, 12.75 ms here), and its collection term divided a floor
arm's whole 15.67 s by 5,400 steps, process startup included - in a window its own note describes as
shared with a live REDQ 1M run. Nothing about that A/B is wrong as a measurement of a 5,400-step run:
1.74x end to end is genuinely what such a run costs, because fixed costs do not scale with the
budget. What it cannot do is extrapolate, which is what the hours column claimed.

The noise is disclosed rather than smoothed: the two with-update slopes came out 17.22 and 16.70 ms,
the two collect-only ones 4.47 and 3.95 ms (`benchmarks/dreamer_real_rate.json`, 2 other CUDA
contexts). A straight line through two budgets is also not `startup + rate x iterations`, because the
first 1,250 iterations of every run sit below `--learning-starts` and carry no update - which is what
made the naive intercept negative (-10.7 s) in the run that first tried this. Solving the same two
budgets against the gate-aware model, the process startup comes back as **5.1 s** without the update
and **6.1 s** with it, so the one-time CUDA graph capture is worth about a second on a budget this
size: cheap, paid once, and invisible in the per-iteration rate.

The batch axis is the second version of a sentence that was wrong before: an earlier window said the
eager update was "flat in batch, sixteen times the samples for free", and the claim was wrong in the
way that matters - it was measured once. On the arm that repeats, batch 16 -> 64 -> 256 costs
**12.62 -> 14.44 -> 22.91 ms**: 1.82x the time for 16x the samples, sublinear and not free. Raw
seconds, per-config medians, the GPU context count of the window and a `diverged_at_rep` marker for
each arm are in `benchmarks/dreamer_update_scaling.json`; no arm diverged in this sweep, and the
marker exists because the heads A/B below does.

**Capture is the CUDA default now.** `--update-graph` is tri-state: unset means capture on CUDA and
eager elsewhere, `--no-update-graph` forces the old path. A `--resume` with no flag follows the mode
the checkpoint was written in, because Adam's state is laid out differently between the two and an
unmarked default would have refused every eager run already in `checkpoints/`; that inheritance is
what makes the flip safe for old runs, and it was checked by resuming a checkpoint written with
`--no-update-graph` on a CUDA box and watching it stay eager. For the same reason the A/B, profile
and reproducibility arms in `bench_dreamer_update.py` now name their mode explicitly - an unmarked
arm would quietly measure the captured path and report it as eager. New runs carry the ~1e-6
Adam-kernel difference described below, on a trainer that is already not reproducible across
processes.

Three obligations come with capture, and all three are tested rather than asserted
(`tests/test_dreamer_graph.py`): the optimizer has to be `Adam(capturable=True, foreach=False)`,
gradients get cleared in place instead of set to `None`, and the two `Normal`s stop validating their
arguments because that check synchronises on the host. The first is the only numerical change, and
the test carries a control arm for it - the same update run eagerly under the capture-only Adam
configuration differs from the default configuration by the same ~1e-6 that the graph differs by, so
the residual belongs to Adam's kernel choice and not to the graph. Checkpoints record which
configuration their Adam state is in (read off `param_groups`, so pre-existing checkpoints answer
correctly) and `--resume` refuses across the two. Replays also have to refresh their noise: a
captured RNG that stopped advancing would imagine the same trajectory forever, which is the failure
mode that test exists to catch.

⚠️ **And Dreamer runs are not reproducible across processes, so "same seed" is not a pairing control
for an outcome claim.** `python bench_dreamer_update.py --mode reproducibility` runs the identical
build twice with the same seed and args and compares: **0 of 42 tensors bit-identical, worst
|Δweight| 7.2e-2 after 100 updates**, and the two runs' 166 episodic returns do not match as a
sequence (`benchmarks/dreamer_reproducibility.json`). Within one process the update *is* exact - the
equivalence test above holds the captured update to 1e-5 on the losses and 1e-4 on the weights
against the eager one with its noise frozen - so this is per-process kernel selection and reduction
order, amplified by 100 updates of a world model that feeds on its own estimates. Timing comparisons
are unaffected, and the KL-batching A/B above is one of those; any Dreamer comparison phrased as
"the same run, but X" is not paired and should be read as two runs.

**Shipped: the two imagination heads as one pass.** After capture there is exactly one launch-level
idea left in the update: `reward_net` and `continue_net` read the same `[h, z]` once per imagination
step, so merging them - shared first layer, block-diagonal output layer - turns six kernels per step
into four at the cost of doubling one tiny matmul's arithmetic. It is the default
(`--heads-impl separate` reproduces the old path). Measured: three independent processes, both arms
in each, alternating, ten timed updates per arm - the fused update is faster in **3 of 3**, by
1.6% to 3.8% (ratios 0.9623, 0.9844, 0.9837; `benchmarks/dreamer_heads_ab.json`). Ten and not
sixty because these arms take real optimizer steps on random weights and random rewards, and the
imagined returns go non-finite after a couple of dozen updates - the paired ratio replicates, the
absolute milliseconds belong to the window.

The equivalence is pinned rather than asserted (`tests/test_dreamer_graph.py`): with noise frozen,
the four world-model losses come out **bit-identical** and only `actor_loss`/`critic_loss` move, by
9.7e-8 and 1.9e-7 relative; the parameter gradients agree to 1e-6; no new parameters exist, so the
checkpoint format is untouched. Writing it also produced a bug the golden test caught immediately:
the `separate` arm wrapped `continue_net` in a second `torch.sigmoid` - the module already ends in a
Sigmoid - which damped the imagined returns and made that arm *look* stable for 60 updates while
measuring a different function. The two same-process comparisons taken before that was fixed
disagreed with each other on sign, and they are discarded rather than corrected: the three
above are post-fix.

**Measured and rejected, because "sounds faster" is not a measurement:**

- **TF32 matmuls: 0.96x** on a full world-model step - no gain, and it changes rounding. These
  networks are launch-bound, not FLOP-bound, so there is nothing for TF32 to win.
- **`nn.GRU` instead of the 49-call `GRUCell` scan:** 2.70x on the scan, but the scan is
  5.18 ms of a ~200 ms step, and the fused layer would not compute the same recurrence here
  (the loop feeds the *sampled posterior* z back into the next timestep). Rejected.
- **Pinned memory in `ReplayBuffer.sample`:** bit-identical and 2.26x on `sample()`
  (0.967 -> 0.427 ms at batch 512), but a REDQ collect step runs `utd_ratio=20` of them inside
  ~480 ms, and the race-free blocking variant gives only 1.22x - about 1% end to end. Not
  shipped: the async form's buffer-reuse race is not worth 1%.
- **Running REDQ and Dreamer concurrently:** 36 and 26 env-steps/s together against ~21 and
  ~16 alone. Both are launch-bound and serialise in the driver, so the sum of throughputs
  drops. Kept sequential.

**Three traps for anyone measuring this.** Rates read off the training logs are wrong: the
trainers print only when an episode ends, so sampling the last `global_step=` line twice a minute
reported 48 env-steps/s for a Dreamer that actually runs at 15.6 - only wall clock over a whole run
is trustworthy. And a long job started with `nohup` from an ordinary shell call is killed along with
that call's process tree some minutes later: three runs died that way in one afternoon, silently,
with no traceback. Both are written up, with what they cost, in
[docs/lab-notes.md](docs/lab-notes.md).

**A third way to lose a run: sharing this laptop's GPU.** The Dreamer 1M attempt of 2026-10-02 got
to 84,456 steps and died with `CUDA error: unspecified launch failure`, inside a window where two
other processes were on the device - and left no checkpoint, because the default interval was 200k
and it had not reached one. Full account in [docs/lab-notes.md](docs/lab-notes.md).

**What was done about it.** `utils/gpu_window.py` reads the window before a run starts, and
`python -m utils.gpu_window` prints it: on this box that is the RTX 4070 Laptop with **371 MiB**
already held by **5** other CUDA contexts - and that is its *idle* state, because Medal, Overwolf
and two Brave renderers keep a context permanently while `--query-compute-apps` reports their
memory as `N/A`. The usable signal is therefore megabytes committed, not the process count. A CUDA
run of **200000 steps** or more is refused above a **1024 MiB** ceiling unless
`--allow-shared-gpu` is passed; shorter runs report the window and start anyway, because what a
contended window costs is the hours lost when a run dies, and a 5k-step test loses minutes. The
window is also logged into the run's MLflow params (`gpu_memory_used_mib_before_run`,
`gpu_other_cuda_contexts`), so a rate can be traced to the contention it was measured in. The
guard is the mechanical half of a habit that predates it: GPU-using measurement jobs are not run
alongside training, and a run that has to survive does not start until the window says it can.

`--checkpoint-interval` on Dreamer went from 200k to 50k on the arithmetic in
`benchmarks/dreamer_checkpoint_cost.json`: one full checkpoint is **6.96 MiB** taking **22.16 ms**
to write, so 20 saves per 1M steps cost **0.44 s** and 139.3 MiB - while 200k was precisely the gap
that left the death at 84,456 steps with nothing to resume from.

**WSL2 does not escape this.** `wsl -d Ubuntu-24.04` reports the same card and the same driver
(`556.29`, `8188` MiB total), so a Linux interpreter buys the CUDA-JAX path that native-Windows
`jaxlib` ships no wheel for - not a quieter GPU.

**The dial that is not free: gradient steps per environment step.** `--utd-ratio` (REDQ,
default 20) and "update every collected batch" (Dreamer with `--num-envs 4`) are how much
learning each environment step buys. Raising `--num-envs` at a fixed `--total-timesteps` keeps
the env-step budget exact and still finishes sooner, but it divides the gradient updates per
sample - a different algorithm, not a faster one. Everything shipped above leaves the env-step
budget *and* the update count alone.

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

**The learner was the other half of the JAX question, and it does not win either.** The physics is
one reason to stay off XLA; the argument for porting anyway was that the learner is a recurrence of
about ten tiny kernels per step, which is exactly what XLA fusion is good at.
`python bench_jax_update.py` rebuilds the imagination rollout in JAX (`lax.scan` + `jit`, the
sampled `z` feeding the next step as `RSSM.transition` does) on the same RTX 4070 Laptop, under
WSL2. Forward only: **1.578 ms** for the 15 steps against the **3.27 ms** the captured torch path
spends on that loop (`benchmarks/dreamer_update_scaling.json`) - 2.1x. But a training step needs
the gradient, and with `value_and_grad` the same loop costs **5.15 ms**: slower than the path it
would replace. Neither is arithmetic - the loop is 195 MFLOP and the card measured
**15.2 TFLOP/s** on a square matmul, a floor of **0.0128 ms** against that 1.578 ms forward - so
what dominates is per-step overhead, which fusion does not delete either. Two caveats that matter:
the torch figure was taken on the Windows host and the JAX one in WSL2, so this is a bound and not
a paired A/B; and the prototype is the rollout, not the world model, the losses or the optimizers.
Together with the table above, there is currently no measured case for a JAX port here.

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
*   **MLOps Tracking:** `mlflow` - every experiment logs its metrics and losses to `mlruns.db`.
    It logs no evaluation scores, and only seven of the thirteen Phase-4 policies archived their
    weights, so a run in that database is evidence a model *trained*, not that it scored; see the
    provenance note under the Phase-4 historical table.
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

⚠️ **Those two numbers were taken on an idle GPU and they do not reproduce under load.** The
same 20k-step budget re-measured while a Dreamer run shared this laptop gave 141.6 s on CPU
and 84.4 s on CUDA = **1.68x**, and `bench_device.py` on that same evening read 27.88 ms (CPU)
against 6.22 ms (CUDA) — 4.48x on the ratio but twice the absolute cost of the idle run, because
the desktop processes and a training peer both queue in the driver. Quote a device factor with
the load it was measured under; `benchmarks/throughput_stacked_ladder.json` records both pairs.

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

`.git` was 1.2 GB while the real history is only ~28 MB: one unreachable, never-pushed 1.15 GB
blob left by a force-added `extratrees_model.pkl` that was later reset. `git prune --expire=now`
plus `git gc --prune=now` cleared it and `git fsck` is clean - the story, with the commands that
found it, is in [docs/lab-notes.md](docs/lab-notes.md).

Three facts to keep in mind:

- `openai_walker/extratrees_model.pkl` (2.24 GB on disk) is **not** version-controlled and
  is regenerable with `python train_extratrees.py`. Do not force-add it.
- `openai_walker/dataset_openai.csv` (74 MB) **is** tracked, on purpose: it is the only copy
  of the Phase-4 dataset, and every table here is measured against it. If the repo ever
  needs to shed it, migrate with `git lfs migrate import --include=...` and a force-push -
  that rewrites published history, so it needs every clone to re-fetch.
- Everything under `checkpoints/` is local and ignored, so it never affected the clone size -
  but 11.4 GB of recovery-run checkpoints were deleted here on 2026-10-01, and what that closed
  off is written up in the lab notes rather than glossed over.

## ✅ Running the tests

The suite is plain `unittest` (no pytest required) and covers the environment contract, the
golden reward rollouts, the parallel/serial vector-env parity, checkpointing and the race
harness — **140 tests, 185 s in this window** (`Ran 140 tests in 184.716s ... OK
(skipped=7)` under `.venv`). Windows of this suite have measured 176.3 s at 102 tests, 269.995 s
at 121, 261.1 s at 127, 329.964 s at 128, 319.168 s at 130, and at 140: 184.716 s, 203.108 s and
306.976 s on three runs minutes apart. The duration belongs to the machine's state, the
count does not, and a gate checks the count so it cannot go stale quietly):

```bash
python -m unittest discover -s tests -t .
```

All seven skips are `tests/test_phase4_behaviour`, which needs `Walker2d-v5` and cannot get it on
gymnasium 0.29; nothing else skips here (torch in `.venv` is a cu121 build and the device answers,
so the CUDA graph-capture test runs). The reverse does not hold: run the same command under
`.venv-phase4` and the suite does not assemble - seven of the eleven test modules fail to import
and seven further tests fail (`Ran 44 tests ... FAILED (failures=7, errors=7, skipped=7)`).
`.github/workflows/ci.yml` is written for that reality: it runs this suite on the root interpreter
and, on the Phase-4 one, only the import check plus `tests/test_phase4_behaviour`.

---

## 📖 Monitoring Experiments

All experiments across Phase 4 (Offline RL) are fully integrated with MLFlow.
To visualize the loss curves, Q-value estimations, and download the archived `.pt` weights
(seven of the thirteen scored policies - `bc`, `iql`, `cql`, the three `bc_sac` variants and
`cql_sac`; BCQ, GAIL, AIRL, DT, MaxEnt and the teacher are not archived):
```bash
# Run this from the root directory
mlflow ui --backend-store-uri sqlite:///mlruns.db
```
Navigate to `http://localhost:5000` in your browser.

**Version requirement: mlflow 3.x, which needs Python 3.10+.** `mlruns.db` is at schema revision
`b7e2c1a4d9f3`, which only mlflow 3.x understands; opening it with mlflow 2.x raises
`Can't locate revision identified by 'b7e2c1a4d9f3'`, and because the trainers' mlflow helper is
deliberately fault-tolerant, that error used to be swallowed - runs looked like they were logging
while writing nothing. `requirements.txt` pins `mlflow>=3.0` for that reason, and
`start_mlflow_run` prints the version and the remedy when it hits the wall.

The history lives in **one** database at the repository root (21 runs, 5 experiments, 252,856
metric rows); `utils/mlflow_uri.py` resolves its path from the repo root, so the working directory
no longer decides where a run goes. Three runs killed mid-training were closed with a
`closed_as_stale` tag rather than deleted, because the metric history is the only surviving evidence
that they ran - see Phase 4, item 6. How each of those states was found, with the commands, is in
[docs/lab-notes.md](docs/lab-notes.md).

## 🔬 Reproducing and measuring

```bash
python -m unittest discover -s tests -t .   # 140 tests in .venv, 185 s; see "Running the tests"
python bench_env.py --seconds 4             # env throughput, physics vs Python split
python bench_mjx.py --sizes 32,128          # MJX/JAX batched stepping
python verify.py                            # Phase-2 artifact check (exits 2 when missing)
python bench_device.py                        # SAC update cost, CPU vs CUDA
python bench_dreamer_update.py --mode scaling # per-loop-step cost, eager vs captured, 3 batch points
python bench_dreamer_update.py --mode loop-split  # one iteration, segment by segment, one window
python bench_dreamer_update.py --mode real-rate   # the shipped command at two budgets, 2 reps/cell
python bench_dreamer_update.py --mode checkpoint-cost  # bytes and ms per Dreamer save, per interval
python -m utils.gpu_window                    # the GPU window a long run would start into
python bench_jax_update.py                    # JAX imagination rollout, fwd + fwd/rev (needs a
                                              # CUDA jaxlib: WSL2, see the MJX section)
```

`verify.py` currently exits non-zero because `dataset.csv` is not in the repository — see the
Phase-2 caveat. That is the accurate answer, not a defect in the checker.
