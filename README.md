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
under them. **Is a million steps enough to walk to the target? No - and no budget measured in this
repository is, under this reward.** Forty times more training on that same run did not produce
walking: the curve is flat from 1M to 40M. And the two Phase-1 runs that did reach 1M on env v9 say
it from the other direction - Dreamer 0 of 50 inside the radius at +0.0055 m/s, ARS 0 of 20 at
-0.0216 m/s - while the task needs 0.2-0.5 m/s held for 250-625 of an episode's 1000 steps. Nothing
measured here is within an order of magnitude of that floor, so the missing thing is not steps.
`walker_recovery_v1` came *first*, 6.4 days before this run and 46-wide against its 49 - it is the
policy the target run continued from, with the widening being the `--init-from-run-id` step (see
"the curriculum here is manual" below for the dates and for what they prove and do not) - which is
the honest context for the Phase-3 result
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
trainer and by `eval_phase1.py` / `evaluate_merging.py`; checkpoints carry their effective
`reward_kwargs` (and their `target_forward_velocity`) so a scorer reads the reward off the artifact
instead of guessing, and `reward_kwargs_for()` falls back to the trainer shaping for every
checkpoint that predates recording it. `--reward-weights env-default` reproduces the old protocol,
and every scored row prints and stores which reward it used.

**That fallback was wrong for three of the four algorithms, and it had already shipped numbers.**
Its first version split the legacy checkpoints by trainer - SAC/TD3/PPO went through
`train_walker.py` so they got the shaping, and ARS/REDQ/Dreamer "never used shaping at all" so they
got the environment defaults. The second half is false. `make_env` has passed
`**TRAINING_REWARD_KWARGS` to every sub-environment since **8d37846 (2026-05-20)**, four and a half
months before any v9 checkpoint existed, and `train_dreamer.py`, `train_redq.py` and `train_ars.py` all
build their environments through it; **ad22c94** (2026-10-02) only moved literals that were already
identical into the shared constant. So every Dreamer, REDQ and ARS return published before
2026-10-04 was computed against a reward those runs never optimised - the defect the paragraph above
describes, on exactly the three algorithms the fix did not cover, and the README repeated it as a
reason ("scored with the environment defaults because `train_ars.py` never used the trainer
shaping"). All three trainers now record `reward_kwargs` in every checkpoint they write, the
fallback resolves to the shaping for the ones that do not, and `tests/test_reward_shaping.py` pins
both halves of the invariant: that `make_env` really does apply every kwarg - which is what makes
the fallback a fact rather than a convenience - and that a legacy checkpoint resolves to the
shaping. The check that it resolved correctly is a pair of artifacts that must agree: the same
checkpoint and seeds scored by default resolution
(`benchmarks/phase1_dreamer_v3_1m.json`) and by the explicit flag
(`benchmarks/phase1_dreamer_v3_1m_training_reward.json`) hold bit-identical per-episode returns
and telemetry.

Re-scoring moved the returns and moved no behaviour, which is the cleanest available evidence that
the shaping is a scoring question and not a training one. On the 1M Dreamer (50 episodes, cuda,
same seeds) the telemetry - steps, closest approach, target reached, x-velocity - is **identical to
the digit**, 19 of the 50 returns changed, and the mean moved 5215.46 → **5315.86** (+1.9%). The
extremes did not move at all (-6998.07 and 14535.79 both before and after), and which episodes moved
is not a matter of degree: all 19 that changed peak at or above **0.650 m** of torso height and all 31
that did not peak at or below **0.647 m**, because every term whose weight the shaping changes is
either multiplied by `standing_gate` or requires the robot to be up at all, and `standing_gate`'s
height factor is `clip((z - 0.65) / 0.60, 0, 1)` - exactly zero below 0.65 m. An episode that never
lifts its torso off the floor scores the same under both reward functions, which is why 31 of 50 did
not move. ARS at 1M moved 8679.40 → **8814.81** with its fall count, closest approach
and x-velocity unchanged at 0.35 / 3.015 m / -0.0216 m/s.

**A second defect fell out of checking the first: the retired tables did not record the device.**
`eval_phase1.py` wrote `protocol`, `env_version` and `env_commit` into its artifact but not the
device it ran on, and the device changes the answer - see "⚠️ The return depends on the device" in
the Phase-1 section below. Five rows turn out to have been scored on **cpu** while the rows they were
compared against were scored on cuda: the 269k step of the budget comparison below, and all four rows
of `benchmarks/phase1_evidence_eval.json`. Both halves of that retired evidence table
reproduce exactly on cpu, each under the weights the retired fallback chose for it:
`benchmarks/phase1_evidence_eval_cpu_defaults.json` returns **3540.76 / 4892.44 / -409.43** for the
REDQ, Dreamer and ARS smoke checkpoints, and `benchmarks/phase1_evidence_eval_cpu_auto.json` returns
**23589.07** for the 40M SAC row - the one row the retired fallback already resolved to the shaping.
The retired 269k Dreamer row reproduces the same way:
`benchmarks/phase1_dreamer_v9_269k_cpu_reproduction.json` returns 6486.27 with median 6176.57 and
0.10 falls on cpu. `eval_phase1.py` now records `device` and
`torch_threads` alongside the protocol, so a row that cannot be compared says so.

A third and smaller one: `--reward-weights training` replaced the whole kwarg set, which silently
dropped the checkpoint's own `target_forward_velocity`. That is a task parameter - the speed the walk
reward is centred on, and the clip on velocity-toward-target - not a reward weight. On the 40M SAC
checkpoint, which records `target_forward_velocity=10.0`, dropping it moved the score from 23589.07 to
23007.95 on the same device and the same seeds: the telemetry is identical and exactly **3 of the 10**
episodes change return (1, 5 and 9 - the ones where the robot moved toward the target), which is what
a clip at 10.0 m/s against 0.8 m/s should touch and nothing else. Both arms are committed
(`benchmarks/phase1_evidence_eval_cpu_auto.json` and
`benchmarks/phase1_evidence_eval_cpu_training.json`), and the second is the only artifact here that
current code will not reproduce, because it was written by the buggy arm; it is kept as the regression
fixture rather than as a protocol. The guard that replaces it is
`benchmarks/phase1_sac_40m_forced_training_control.json`: forced and auto resolution must now agree on
that checkpoint, and on cuda they both give 17363.56.

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
(`benchmarks/phase1_ars_v2_1m_v9.json`, 20 seeded target-phase episodes on cuda, scored with the
trainer shaping it was actually trained under): mean 8814.81, median 8937.07, std 5059.09, 0.35
falls per episode, **0 of 20 inside the radius**, mean closest approach 3.015 m, mean x-velocity
-0.0216 m/s. Same verdict as v8's ARS at the same budget, which is the comparison that makes the v9
change safe to have made: the reward shape moved the returns, not the behaviour. This row was
published once already as 8679.40 under the environment defaults, on the false premise that
`train_ars.py` never used the shaping; the three behaviour columns are identical in both versions,
because the reward weights do not touch the trajectory.

The second Phase-1 run to reach its full budget on v9 is **Dreamer at 1,000,000 steps**
(`benchmarks/phase1_dreamer_v3_1m.json`, 50 seeded target-phase episodes on cuda, `run-id
dreamer_v3_1m`, seed 7, capture on): mean 5315.86, median 5581.13, std 5150.85, min -6998.07,
max 14535.79, 0.14 falls per episode - and **0 of 50 inside the radius**, mean closest approach
2.893 m, mean x-velocity 0.0055 m/s, standing at the end of the episode 0%. So the run completes,
and the verdict is the same one the ARS run and the inert-reference table already give: the return
is the posture bonus, and the animal does not walk. It is a small favour to the reader to say that
plainly instead of letting 5315.86 read like a score.

Two useful comparisons fall out of it. Against the *partial* v9 checkpoint - the run that stopped at
269,404 steps and is recorded in `benchmarks/phase1_dreamer_v9_269k.json` - the full budget moved
the mean from 6368.15 down to 5315.86 (-16.5%) and the std from 4272.68 up to 5150.85, on 50
episodes rather than the 10 the partial one was scored on: **more training did not buy a better
policy here**, which is the honest reason to stop spending 1.27 h runs on this task until the reward
makes walking attractive. That sentence used to read "from 6486.27 down to 5215.46", and both of
those numbers were wrong in a way that matters more than the 2% they are off by: 6486.27 was scored
on **cpu** and 5215.46 on cuda, so the comparison was between two devices as well as two budgets
(`benchmarks/phase1_dreamer_v9_269k_cpu_reproduction.json` reproduces 6486.27 exactly on cpu, which
is how the row was identified). Both rows are now cuda, both are the training reward, and the
direction of the comparison survives the correction. And against ARS at the same budget, Dreamer
sits closer to the target on average (2.893 m against 3.015 m) with a lower return - two different
ways of not reaching it.

What the task geometrically requires is not in dispute: `timestep=0.002` with `frame_skip=5` makes one env step 0.01 s, episodes are capped at 1,000 steps (10 s of simulated time), targets
spawn 2-5 m away and the success radius is 0.45 m. Reaching the near target needs 0.2 m/s
sustained, the far one 0.5 m/s; at the env's own nominal 0.8 m/s the walk itself is 250-625 env
steps of a 1000-step episode. Every checkpoint above averages ~0.0 m/s, so none of them is
anywhere near that floor. What a retraining costs in wall clock is no longer a projection for
Dreamer: the run above took **1.27 h** of machine time for its 1M steps (`benchmarks/dreamer_real_rate.json`
measured 1.18 h before it ran, and the run itself came out 8% slower), and the throughput section's
"What a training run costs" carries the SAC-stack figures.

**And the rung below walking is the one nobody clears.** The environment is a curriculum -
`standup_balance_walk_curriculum_v9` - so "does it walk?" presupposes "does it stand?", and the env
defines standing as torso height inside `healthy_z_range = (1.0, 2.0)` m. `python bench_posture.py
--episodes 50` (`benchmarks/phase1_posture_probe.json`) measures the height instead of inferring it
from the return, on 50 seeded episodes per row (the ARS score above used 20, so its return here is
not the same number), on the device the published harness
resolves to (cuda) - which is why its return column reproduces `5315.86` to the cent:

| model | episodes that ever reached the band | mean peak torso z | % of steps in the band | falls / episode | mean return |
|:---|---:|---:|---:|---:|---:|
| SAC at 40M steps | 23 of 50 | 0.864 | **14.73%** | 0.46 | 24534.17 |
| SAC at 9M steps | 28 of 50 | 0.947 | 10.39% | 0.72 | 20447.84 |
| ARS at 1M | 16 of 50 | 0.883 | 0.84% | 0.38 | 8560.92 |
| Dreamer at 269,404 | 10 of 50 | 0.713 | 0.68% | 0.20 | 5572.43 |
| **Dreamer at 1M** | **7 of 50** | 0.701 | 0.19% | 0.14 | 5315.86 |
| commanding zero | 7 of 50 | 0.557 | 0.66% | 0.14 | 2964.48 |
| uniform random | 7 of 50 | 0.567 | 0.46% | 0.14 | 2172.91 |

Two rows of the first version of that table were wrong, and both errors are the kind a return cannot
reveal. **(1) The row labelled "SAC at 40M steps" was the 9M checkpoint.** The script's roster held
one entry keyed `sac_40m` pointing at `sac_ckpt_9000000.pt`, so the published 28 of 50 / 10.39% /
20447.84 measured `sac_ckpt_9000000.pt`; the real 40M policy is the row above it, and it stands for
14.73% of its steps while entering the band in *fewer* episodes (23 of 50) - it gets up less often
and stays up longer. Both rows are now in the table, every row records the checkpoint path it was
scored from, and a gate compares that path's step count against its label. **(2) The passive
references were scored under a different reward from the trained rows.** `commanding zero` and
`uniform random` have no checkpoint to read a reward off, so they fell back to the environment
defaults while every trained row used the shaping - two reward functions in one column, and the
defaults pay `standing_reward=50/step` that training sets to 0, i.e. the yardstick was paid for
lying still. All rows now use `TRAINING_REWARD_KWARGS`. One caveat the weights do not cover: a row
also carries the `target_forward_velocity` its own checkpoint records, so the table spans three task
speeds - the environment's 0.8 for the two Dreamer rows, ARS and both passive references, and the
10.0 and 1.2 that the 40M and 9M SAC runs were trained at. The posture columns are unaffected, and
the comparison reading 3 rests on, Dreamer against commanding zero, is between two rows on the same
0.8. The control that identifies
which numbers moved is the one row whose weights did not change: SAC at 9M was already shaped and is
the only row
whose return is identical to the cent (20447.84), while the zero reference dropped 3202.71 →
2964.48 once the standing bonus was removed. Every height, fall and band column in the table is
unchanged by all of this, because none of them is computed from the reward - the environment-default
arm of the same probe (`benchmarks/phase1_posture_probe_env_default.json`) reproduces all five
retired returns exactly (5215.46, 5476.97, 8340.83, 3202.71, 2130.14) with every geometry column
identical to the table above.

Three things come out of that table, and one of them corrects a sentence higher up on this page.

1. **The 1M Dreamer reaches standing height in as few episodes as commanding zero - 7 of 50 each -
and spends less time up than a policy that emits nothing at all** (0.19% of steps against 0.66%). "The
animal does not walk" was true and far too generous: by the environment's own gate it mostly does not
stand either.
2. The fall counts are why nobody noticed. `terminated` latches only after an episode has been
healthy once, so the trained policy and doing nothing report the **same 0.14 falls per episode** - a
low fall count in this env measures the latch, not balance, and it was read as stability.
3. The ordering is nearly the honest one, and where it is not, it is not in the trained policies'
favour. Return and time-on-feet agree at the top (both SAC rows) and at the bottom (uniform random
is last on both), but the 1M Dreamer **outranks the zero-action reference on return while standing
for a third as long** (5315.86 against 2964.48, 0.19% of steps against 0.66%): the return is a
posture proxy that a policy can beat without standing, which is the whole reason this script exists.

⚠️ **The return depends on the device the evaluator runs on. The posture *verdict* survives that;
the posture *numbers* do not always.** Same
checkpoints, same seeds, cuda against cpu (`python bench_posture.py --compare-devices`,
`benchmarks/phase1_eval_device_sensitivity.json`): the 1M Dreamer scores **5824.49 on cuda and
4737.23 on cpu**, 20.6% apart on the same ten episodes, and the 269k one 6368.15 against 6620.16.
The two entries that do *not* move are ARS at 9024.24 on both devices and the zero-action reference
at 2829.82 on both. The first version of this paragraph explained that by saying neither action
"depends on a sampled latent", and the SAC row refutes it: `agent.get_action(...,
deterministic=True)` has no sampling anywhere, and the same checkpoint under the same reward scores
**23589.07 on cpu against 17363.56 on cuda** - 30% apart, with 7 of the 10 episodes on
completely different trajectories (episode 1's closest approach 1.917 m on cpu, 3.608 m on cuda).
The mechanism is not stochasticity, it is that this is a chaotic system scored in float32: any
rounding difference between the two devices' matmuls compounds over a thousand steps, and a deep MLP
amplifies it while ARS - a single linear map evaluated in float64 numpy, identical on both devices -
and the zero policy, which does no arithmetic at all, have nothing to amplify. Dreamer's reparametrised
posterior adds a second source on top. The standing verdict is
unaffected for the run this section is about (0.12% of steps in the band on either device for the 1M
run, 1 of 10 episodes reaching it on both), which is the point of measuring height instead of reading
it off the return - but the posture *figures* are device-dependent too wherever the policy diverges:
the 269k checkpoint reads 0.86% of steps in the band on cuda against 0.42% on cpu, 3 episodes against
1. Both readings are "nowhere near standing", so the conclusion holds; a posture number quoted to two
decimals without its device does not. That is why this script pins the device
and the torch CPU thread count into its artifact, why `eval_phase1.py` now records both as well, and
why a Phase-1 return quoted without them is
only half a measurement. (The thread pin is not decoration either: with torch's default threading the
identical command gave 0.634 m and 0.616 m peak torso height on two runs of the same checkpoint.)

**Would one run that does everything beat the phase split? The target phase already is that run.**
`task_phase=target` is not "walking only": its reward is the constant 1.0, the v9 posture bonus graded
from the floor up, the recovery and upright terms, the stability term, progress toward the target,
velocity toward the target, perpendicular-drift penalty and the success bonus - the `reward = (...)`
sum in `WalkerRagdollEnv.step` - and it is the only phase the trainers give
`terminate_when_unhealthy=True` (`train_walker.env_common_kwargs`). Every number in this section - ARS
at 1M, Dreamer at 250k, 269k and 1M, the whole posture table - comes from a single run on `target`
starting from `reset_mode=mixed`, which is 25% fixed-fallen, 15% upright and 60% randomized-fallen
resets. So "train once and do everything" is not a missing feature here; it is what has been measured,
and the 40M SAC policy is the only thing that ever cleared the standing rung on it (14.73% of its steps
in the band). The 1M Dreamer is the same unified task with a smaller budget and a different algorithm,
and the difference between those two rows is 14.73% against 0.19% - not a difference in task setup.

**One phrase in that paragraph was doing more work than it could carry.** The 40M SAC policy it names
is `walker_target_v1`, and the manual curriculum seeded it from the recovery expert
(`benchmarks/curriculum_provenance.json`): that is one run on the unified task, but it is not a run
from zero on it. The arm that was missing now exists - `sac_target_from_scratch_40m`, seed 7,
`num_envs=32`, `target_forward_velocity=1.2`, `reset_mode=mixed`, no `--init-from-run-id`, 40M steps
in 5 h 11 min 37 s at 2,145 env-steps/s sustained. Its ten checkpoint budgets are compared with the
curriculum arm's ten checkpoints **per seeded episode**, not as two means, in
`benchmarks/target_learning_curve_from_scratch_paired.json`.

Re-scoring the curriculum arm was not ceremony. Its committed curve predates `eval_phase1.py`
recording the device, and the same checkpoint scored cpu against cuda is not the same measurement -
but here all 200 telemetry rows came back identical to that artifact
(`curriculum_rescore_matches_published`), so this arm is device-insensitive and the pairing holds
either way. Both columns were run in one session, `cuda`, 24 torch threads, env v9.

| Over the 200 matched episodes | from scratch | curriculum (`walker_target_v1`) |
|:---|---:|---:|
| episodes that got closer to the target | 154 | 38 (8 pairs within 5 mm) |
| median closest approach | 2.367 m | 3.075 m |
| best single episode | 0.038 m | 0.313 m |
| episodes ending inside the radius | 17 | 2 (never the same episode) |

Median advantage 0.527 m, mean 0.768 m. So the phase split is not what was keeping the agent from
walking to the target: starting from zero on the unified task does it more often, and no later. What
the extra budget did not buy is reliability. Reach rate by checkpoint (1M, 2M, 3M, 5M, 8M, 10M, 15M,
20M, 30M, 40M) is 5.0, 0.0, 0.0, 20.0, 15.0, 5.0, 10.0, 15.0, 15.0, 0.0 percent - the run's last
checkpoint is the least goal-directed of its second half and its best is at 5M. A longer budget moves
this curve; it does not straighten it.

**Every reach number above and below is distance-only, and a rendered episode shows why that matters.**
`reached_target_pct` counts an episode whose closest approach ever entered the 0.45 m radius. The
environment's own success term is stricter - it also requires torso height above 1.0 m and upright
above 0.7 *at that step* - so a fall forward can land inside the radius and be paid nothing. Replaying
the scored episodes into video (`bench_reach_definitions.py` re-scores them under both rules and
refuses to write unless the distance-only column reproduces the committed artifact episode for episode)
showed exactly that: the clip that "reached" at 0.299 m got there at a step with the torso at 0.405 m,
and it ends 1.23 m from the marker, motionless, 7.5 s later. Under the strict rule, the best arm in the
published world for everything published before the reward-override experiment arrives standing in
**3 of 20 episodes (15%)** - SAC at `--utd-ratio 4` seed 7 / 3M and at ratio 1 seed 8 / 4M, whose
distance-only rows read 25% and 20% - and the 40M from-scratch curve's 20% at 5M is **10%** upright. PPO
at 10M is the extreme: 25% and 15% by distance, **5% and 0%** standing, because four of its five arrivals
are collapses. The one arm among them where every arrival was clean is the `fast` world at 5M seed 8 (20%
by both rules, 0 collapses), which is a reminder that the preset changes what arriving means, not just how
long it takes.

**An upright arrival says the body was standing at that moment; the step trace says it still was not
walking.** Splitting every step of the same published episodes at whether the torso was in the band then
- `bench_approach_mechanism.py`, which replays all 460 scored episodes of the 23 quoted checkpoints and
refuses to write unless each one reproduces its committed telemetry - the 35 episodes the distance rule
counts as reaches in the published world close **46.61 m at in-band steps against 43.43 m with the torso
below the band** (51.8% of the ground), and **30 of those 35 closest approaches come after the episode's
first fall**. The standing time is not what is short; the continuity is. Those episodes spend 12.7% of
their steps in the band but never more than **1.47 s** at a stretch, and no reaching episode in the
published world ever stands for two continuous seconds - which is what the clips look like. What survives
is direction: at in-band steps the body moves toward the target at 0.34 to 1.58 m/s depending on the arm,
and the best near miss in the repository comes to 1.8 cm inside the radius.

**So the experiment was to change the gait, and one of the two arms did.** Both run at the dose the last
two nights fixed (5M steps, `num_envs=32`, `--utd-ratio 4`, two seeds each), and each changes one thing
the target phase actually computes. One candidate died before it was run: `stillness_penalty` is only
computed when `task_phase == "balance"` (`envs/walker_ragdoll_env.py:493-498`), so raising
`stillness_penalty_weight` in the target task changes no gradient at all - which is what the first draft of
this experiment proposed. The live "do not collapse" term in target is `stability_reward_weight`, so arm A
triples it, 20 to 60, through the new `--reward-override`; arm B is the new `--target-curriculum`, targets
at 2.0-2.5 m opening 0.5 m every three successes.

Arm A is the first thing in this repository to move the strict column: **4 of 20 episodes (20%) arrive
standing** at its 4M checkpoint on seed 7, where 3 of 20 had been the ceiling, and its seed 8 best cell
lands back on 3 of 20 - the direction replicates, the peak does not, and both seeds fall off by 5M. Arm B
beats nothing: 1 of 20 upright at its best in-task cell and 1 of 20 when the same checkpoints are scored on
the published 2-5 m, having spent 4.35-10.86% of its steps in the band against the published arms' 12.7%
and falling 0.6 to 1.5 times an episode against arm A's 2.05 to 2.6 per episode.

The step trace says arm A bought more than a lucky arrival frame. Across its three quoted cells the reach
episodes close **49.4% to 73.5% of their ground inside the band** against the published pool's 51.8%, they
are in the band for 13.7-17.1% of their steps against 12.7%, and the best cell's upright arrivals hold the
band for 1.17 s at a stretch. What it did not buy is the thing the paragraph above is about: over all
**60** reaching episodes now traced in the published world, the longest continuous stand is still **1.47 s**
and none of them reaches two seconds.

**A planner with simulator access reaches the marker more often than any learned arm, and still does not
walk.** `bench_mpc.py` scores a receding-horizon cross-entropy planner on the same 20 seeded episodes,
against the same environment and the same shaped reward the learned policies optimise, and measures it
with the same per-step trace. At a small search budget (24 samples × 2 CEM iterations, 0.6 s horizon) it
reaches in **3 of 20** and arrives standing in **3 of 20** - the published ceiling. Drawn a second time it
is identical on every column (reach, upright arrivals, band time, share closed standing), which matters
because the constraint solver is not bit-deterministic: two rollouts from a restored state diverge within
a few steps, so the planner searches an approximate model, and the closed loop is nevertheless
reproducible. Raising the search (48 × 3, 1.0 s horizon, replanning every 0.2 s) puts it at **9 of 20 by
distance and 6 of 20 standing (30%)**, above the stability arm's 4 of 20 - and its gait gets no better:
it closes **27.9%** of its ground inside the band where the published pool closes 51.8% and the stability
arm 71.3%, and its longest continuous stand is 1.40 s against the learned arms' 1.47 s. It also costs
67 s and 157 s of one CPU core per 10 s episode (288k and 720k simulated steps), against a 5M-step SAC
run that costs ~2,000 s of GPU once and then acts for free.

So the residual-learning question now has a measured target rather than a hypothesis. Search already buys
arrival; what neither a planner with the true dynamics nor a policy with 5M steps of experience buys is
closing ground *on its feet* - and across every cell measured here, learned or planned, nothing holds the
standing band for two continuous seconds.

**Cloning the planner fails, and it is the cloning that fails.** `collect_mpc_demos.py` recorded the
teacher's own trajectories on seeds 101-128 - disjoint from the 11-30 every published row is scored on, so
a student cannot memorise its exam - and there the teacher reaches in **12 of 28** and arrives standing in
**6 of 28**, with 12.4% of its steps inside the band. `train_bc_ragdoll.py` then trained two students on
`SACAgent`, the same network the SAC rows use, one imitating every recorded step and one imitating only
the in-band steps (3,477 of 28,000 transitions). Both score **0 of 20** on the published protocol, at mean
return 2,941.05 and 841.61. `bench_bc_teacher.py` says why, on one episode-level split where every
predictor is scored on the same held-out rows: predicting a constant zero action gives validation MSE
**0.3753**, the per-dimension mean 0.3751, ridge 0.3758, nearest-neighbour 0.750 at k=1 falling to 0.3867
at k=32, and the trained students 0.3730 and 0.3920 - nothing recovers any state-dependence from the
49-wide observation. The reason is in the teacher's own signal: consecutive recorded actions differ by MSE
**0.7567** against the action's own energy of 0.3806, because the executed action is the winner of a
stochastic search and flips between near-tied sequences step to step. Averaging the elite set instead of
taking its argmin does not fix that and costs arrival: on a 4-episode probe the jitter moves from 0.7567 to
0.7408, ridge (0.4039) and nearest-neighbour (0.4072) still lose to the constant predictor (0.3952), and
the teacher drops to 1 of 4 reaching with no upright arrival. So what the planner knows is not expressible
as a policy over this observation - which is the honest obstacle in front of the residual idea, not the
search.

**What the planner knows is predictable, even though what it does is not.** The upgraded collector also
records the teacher's own lookahead value per step - the return of the best sequence it found from that
state - and the same episode-level probe against that scalar, on the first 8 collected episodes, gives
ridge R² **0.4738** on held-out episodes, where restricting the regression to the three target components
of the observation gives **−0.0677**, no better than the mean. So the value is not the distance-to-target
feature re-derived; it carries posture, contact and velocity. That is the one property of the planner that
survives the observation, and it is what the next paragraph tests.

**The value transfers to a regressor; on this data it does not transfer to a policy.** Re-measured on the
full 28-episode preference set, ridge predicts the teacher's lookahead value with R² **0.7722** - up from
0.4738 on the 8-episode probe that motivated the route - while the three target components of the
observation alone give **-0.0013**, so the signal is state and not goal geometry.
`train_offline_critic.py` fits a twin critic by TD on those same transitions and ascends the actor through
it, with and without a TD3+BC in-distribution term. Both students reach in **0 of 20** and arrive standing
in 0 of 20, at mean return 1,013.50 and 1,013.32, spending **0.22%** of their steps inside the standing
band where the teacher spends 12.64% - and their executed actions are further from the teacher's (MSE
1.3736) than a constant zero (0.3774), because ascending a learned Q saturates the outputs. The training
curve locates the failure: held-out TD error grows from 4,949.97 to 240,268.17 over 60 epochs, so the
checkpoint validation selects is from epoch 5, before the actor has anything to exploit, and at that epoch
the two actor modes are indistinguishable. So the planner's judgement is learnable as a critic and, trained
offline against that critic, not usable as a gait. The route this leaves untested is putting the learned
value inside the online loop as an auxiliary signal, rather than training a policy on planner transitions
alone.

So the honest answer to "does an agent walk to the target?" is: **the target is in the behaviour and the
gait is not.** These policies head for the marker whenever they are up, arrive standing in about one
episode in five at best, and get there by rising, advancing for under a second and falling again - not by
walking the 2-5 m that separates them from it. Paying three times as much to hold a stable standing pose
is the one lever measured so far that moves both the arrival column and the share of ground closed on
feet; giving the controller a simulator to search in moves arrival further still (6 of 20) and the gait
not at all. Making the stand last longer than 1.5 s is the open problem.

The standing rung splits the other way, and only the instrument that measures the torso can say so.
On `bench_posture.py` at the settings the published posture table uses (50 seeded episodes, seed 11,
`cuda`, torch threads pinned to 1) the curriculum row reproduced its committed number exactly -
14.73%, 23/50 episodes ever in band, mean peak 0.864 m - which is what licenses the three rows under
it.

| checkpoint | % of steps in the standing band | episodes ever in band | mean peak torso z | best peak |
|:---|---:|---:|---:|---:|
| from scratch, 8M | 9.73 | 39/50 | 1.387 m | 2.069 m |
| from scratch, 20M | 8.43 | 35/50 | 1.357 m | 2.160 m |
| from scratch, 40M | 8.65 | 36/50 | 1.317 m | 2.056 m |
| `walker_target_v1`, 40M (control) | 14.73 | 23/50 | 0.864 m | 1.525 m |

Neither arm wins this criterion; they answer different questions about it. The curriculum arm holds
the band for a larger share of every episode (14.73% against 8.43-9.73%), but its *mean* peak torso
height, 0.864 m, does not reach the band's floor at 1.0 m and it gets into the band in 23 of 50
episodes. The from-zero arm is in the band in 35-39 of 50, peaks at 1.317-1.387 m on average - above
the floor - and reaches 2.160 m at best. "The only thing that ever cleared the standing rung" was
true of the arms measured when it was written; measured against a from-zero run at the same budget,
the honest statement is that the arm fine-tuned from a recovery expert stands for longer when it
stands, and the arm trained from zero on everything gets up more often and higher. Neither does both.

One caveat that applies to both tables: the episode *returns* of the two arms are not comparable,
because the curriculum checkpoint records `target_forward_velocity` 10.0 and the from-scratch
checkpoints record 1.2 - two reward functions for one task speed. Closest approach, reaching the
radius and every torso measurement do not depend on the reward, which is why the comparison is built
out of those and not out of returns.

What the phase split does cost is that a phase cannot change mid-run. `task_phase` is read once in the
constructor and decides the observation width - `observation_size = 49 if task_phase == "target" else
46`, the three extra components being the target's relative x, y and clipped distance - and that width
sizes the networks, the replay and sequence buffers, the observation normalizer, the PPO rollout
tensors and, for Dreamer, the static buffers inside the captured CUDA graph. A run that promoted itself
from `balance` to `target` would invalidate all five. So this repository composes skills *after*
training instead: Phase 3 merges separately trained
policies (task arithmetic in `merge_models.py`, an MoE gate over the 46-wide observations in
`train_moe_gate.py`), and `train_walker.py --init-from-run-id` starts a new run from another run's
actor, zero-filling the three new input columns (`load_state_dict_with_expanded_input`) and padding
`obs_rms` (`adapt_obs_rms`). That is a phase curriculum across runs, which the 46→49 widening permits;
it is not one run, and the difference is that the replay buffer and the normalizer statistics do not
survive the switch.

The in-environment curriculum that would let a single run get progressively harder is the part that
is unused: `target_curriculum_streak` is accepted and forwarded to `EzPickle` for re-pickling but no
attribute of the env ever reads it, `target_success_streak` is hard-coded to 0 in the step info dict,
and `_target_fixed_until_curriculum` is assigned once in `__init__` and never read
again. The only adaptivity that actually runs is target resampling - 2-5 m away, within ±0.15 rad, on
every reset and on every success, with `_curriculum_level` counting the successes.

**That is not the same as the curriculum being absent: in this project it is manual, and the
pre-existing policies are its product.** The checkpoints say so, and this paragraph was written after
the claim above was corrected by the person who ran it. `walker_recovery_v1` holds 2 checkpoints of a
**46-wide** policy stamped `env_version = standup_recovery_resets_v3`, written 2026-05-11 09:55 (1M
steps) and 2026-05-12 18:11 (20M). `walker_target_v1` holds 41 step positions of a **49-wide**
policy stamped `standup_balance_walk_curriculum_v4` with `task_phase=target`, 2026-05-17 19:28 (1M)
through 2026-05-18 19:00 (40M). So recovery precedes target by 6.4 days, and the widening is exactly
the 46->49 step that `--init-from-run-id` exists to carry across:
`load_state_dict_with_expanded_input` copies the overlapping columns and zero-fills the three new
ones, and `adapt_obs_rms` pads the normalizer statistics. The staging is not only across runs either -
inside `walker_target_v1` the nominal walking speed recorded in the checkpoints is **1.2 m/s at the
13,629,184-step file (2026-05-17 22:29:58) and 10.0 m/s at 14,000,000 (2026-05-18 11:32:58)**: a
night's gap, then a restart with a different task parameter. A phase, in this project, is something a
person changes between runs or at a restart; nothing in the environment promotes it by itself.
`python summarize_curriculum_provenance.py` rebuilds all of it, from the checkpoint metadata alone,
into `benchmarks/curriculum_provenance.json`.

What the dates do and do not establish is worth stating precisely. They establish order: the recovery
files are 6.4 days older and 46-wide against 49, so they cannot have been fine-tuned from weights that
did not exist yet. They do not establish that target was initialized *from* recovery - that would need
a weight-level comparison the metadata cannot give. And they contradict the two-step story this README
used to tell in Phase 3 ("first we pre-trained `walker_target_v1`... then `walker_recovery_v1` was
fine-tuned from it"), which is the sequence reversed; Phase 3 now points here.


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
1. First came a base agent trained to get up: `walker_recovery_v1`, 20M steps, its checkpoints
   written 2026-05-11 and 05-12, its actor **46 wide** and stamped `env_version` v3-era.
2. Then came `walker_target_v1` - 40M steps, checkpoints 2026-05-17 and 05-18, actor **49 wide**,
   v4-era - trained toward the target phase, and reaching the 0.45 m radius twice in 200 seeded
   episodes at a median forward speed of ~0.00 m/s (see the Phase-1 section for that measurement,
   and "the curriculum here is manual" in the same section for the ordering and what the timestamps
   do and do not prove). This README used to say the two steps in the opposite order - that the
   walking agent was pre-trained first and recovery was fine-tuned from it - which the file dates
   contradict; the arithmetic below is what was actually computed, and the benchmark numbers are
   unaffected by the correction to the prose.
3. Because the two policies share an initialisation and a geometric parameter space, algebraic
   operations on their matrices are meaningful.

**Merging Techniques Evaluated:**
- **Task Arithmetic (`merge_models.py`):** Subtracts the target-phase weights from the recovery
  weights to isolate a task vector ($\tau = \theta_{rec} - \theta_{walk}$, in that order - note that
  on the corrected chronology this is the *negative* of the base-to-derived displacement), then adds
  it to a policy.
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
20-episode run at seed 123 gave the same ordering, so the picture is stable. The GAIL retrain below
was scored twice more under this protocol and kept in
`openai_walker/final_episodes_50ep_seed2026_gail_{retrain,june_control}.json`, summarised together
with both runs' training-side episodes into `benchmarks/phase4_gail_retrain.json`. Regenerate with
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
window nothing records what it held. It is listed among the unsupported figures above. What it
lacked was a candidate cause - BC's retired 3837.80, by contrast, is +0.47 sigma inside its own
distribution and needed no explanation at all - and a second run of the recipe below supplies one.

Two more candidates have since been closed by reading rather than by measuring. It is not a
comparison across two policies: `evaluate_all.py` loads `gail_model.pt` straight out of
`openai_walker/`, and that is the file the 50 seeded episodes above scored. And it is not a reward
bookkeeping difference either - `train_irl_gail.py` accumulates its episode return from the
environment's own reward (`next_state, reward, terminated, truncated, _ = env.step(action)`, then
`episode_reward += reward`) and prints and logs it under the name it was scored by:
`True Env Reward:` and the `true_env_reward` metric. The discriminator's learned reward never
reaches that number, so the retired figure and the seeded protocol measure the same quantity. What
is left is the twenty-minute window in which nothing records what the file held, or a hand-typed
figure.

**A second 1M-step run of the same recipe puts the gap in perspective.** `train_irl_gail.py` seeds
nothing, so this is a fresh draw rather than a reproduction: 1,000,000 steps, batch 256, 3 h 18 min,
scored at the published protocol (50 episodes, seed 2026) with the June weights re-scored in the same
session as the control. The control reproduced the published row exactly - mean 998.07, std 0.37,
min 997.31, max 998.74 - so nothing in the harness moved between the two captures. The fresh draw
did not reach the June policy's level.

| GAIL policy (1M steps each) | Eval mean | Eval std | Worst / best episode | Best training episode |
|:---|---:|---:|---:|---:|
| June 2026, `gail_model.pt` | 998.07 | 0.37 | 997.31 / 998.74 | 1092.62 |
| Retrain, 2026-10-05 | 979.35 | 2.30 | 971.41 / 981.95 | 1001.56 |

Two things follow. The first is arithmetic: 1016.41 is **34.46 above the best episode the retrain
produces**, and 17.67 above the best the June policy produces, so a second independent draw of the
recipe does not reach it either. The second is the candidate cause. `true_env_reward` - what the
trainer prints per completed episode - is not the seeded evaluation mean, and the June run's own
episodes reached 1092.62: **8 of its logged episodes sit at or above 1016.41**, the nearest being
1016.53 at step 593,171, which is 0.12 away. So the retired figure is exactly the size of a number
copied off the training print of the run that produced the checkpoint, which is a quantity this
protocol never reports as a score. It is not that number itself: no logged episode is 1016.41, and
the retrain's best episode (1001.56) and its last-20-episode mean (870.31, against June's 902.18)
both sit below it. The cause is therefore attributed, not identified.

This also refines what "near-zero variance" measured. That is variance *within one policy* over 50
resets of fixed weights. Across two runs of the recipe the mean moves 18.72 points and the band goes
from 1.43 to 10.54 - 7.37 times wider - so a plateau "near 1000" has a run-to-run spread of its own,
and one GAIL number carries both.

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
  a bad evaluation, is what its near-zero variance measures - and the plateau is per run: the second
  1M-step draw above sits at 979.35 where this one scores 998.07.
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
**1.17x and 1.40x** on an idle machine, not 1.5x and 6.5x — see "The whole stack, end to end" below,
where the single contended draw that first said 1.22x and 1.86x is reproduced and corrected.

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

**The envs-per-worker curve, with the same treatment** (`python bench_env.py --sweep --n 32
--seconds 4 --reps 3 --json benchmarks/vec_backend_scaling_reps3.json`, one pool rebuilt per cell,
three reps each): one environment per worker is the fastest grouping and every grouping above it
costs throughput.

| `envs_per_worker` | median env-steps/s | rep spread |
|---:|---:|---:|
| 1 | **18,902** | 2.0% |
| 2 | 17,484 | 0.5% |
| 3 | 14,463 | 10.7% |
| 4 | 14,653 | 16.4% |
| 6 | 12,224 | 0.2% |
| 8 | 10,518 | 1.0% |

The 3 and 4 cells are inside each other's spread, so the honest reading is "flat from 3 up, and
monotone after that". What is worth keeping next to it: this same configuration measured 18,902
env-steps/s here and 11,242 in the morning replication of the same command - **68% apart**, with a
0.2-2% within-curve spread here against the 50% the replication showed. Tight reps make a window
reproducible; they do not make it transferable.

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

⚠️ **That table was a single draw on a contended machine, and the replication says less.** Its own
artifact records "a Dreamer 1M-step run was training concurrently on the same laptop for the whole
session". `python bench_stacked.py --ladder n32 --reps 2` re-ran it after that trainer exited
(`benchmarks/stacked_sac_n32_reps2.json`), and both the times and the ratios moved:

| 600k steps, 32 envs, idle machine | rep 1 / rep 2 s | median | env-steps/s | vs committed |
|:---|---:|---:|---:|---:|
| original env + sync | 211.1 / 205.2 | 208.1 | 2,883 | — |
| rewritten env + sync | 179.9 / 177.0 | 178.4 | 3,363 | **1.17x** |
| rewritten env + parallel | 148.1 / 148.4 | 148.2 | 4,049 | **1.40x** |

The two reps agree to 2.9%, 1.6% and 0.2%, so this is not noise hunting: the *committed* arm was
2.6x faster when the box was free, and the parallel arm's advantage shrank from 1.86x to
**1.40x**. Contention does not hurt every arm equally - a 32-environment `SyncVectorEnv` starves in
one thread while 32 worker processes each keep their own - which is how a shared machine managed to
*overstate* the parallel backend. The startup figure moves with it: the sentence below used to
attribute ~200 s of the 290.9 s to booting the workers; against the replication's own steady rate
the boot cost is ~95 s of the 148.2 s.

As the contended artifact's own note reads it: `bench_env.py` reports 7,328 env-steps/s steady state
for that last line, while over 600k steps the same configuration delivers 2,062, because ~200 s of
the 290.9 s is 32 child processes importing torch and mujoco; from those two rates it extrapolated a
1M-step collection at ~2.1x over the committed sync backend. **That extrapolation is not supported
by the replication** - 1.40x is what was measured end to end, and the ~200 s startup belongs to a
window in which the worker processes were also competing with a training run.

**What the repo's own 40M-step run cost.** `checkpoints/walker_target_v1/` holds 41 actor
checkpoints of one real run at `num_envs=32`; their mtimes date the run itself: 37.6M env steps
in 547 min of continuous training = **14.5 min per 1M steps (1,146 env-steps/s; segments range
10.5-16.7 min)**, spread over 23.5 h of calendar time because of two pauses. That is the same
shape of run as the 1,109 env-steps/s line above, which is the cross-check that makes the
summary of this section credible: **a committed training run gets ~1.6x faster at 8-env budgets
and 1.40x when collection dominates, measured twice on an idle machine (1.9x in the single contended
draw) — not 6.5x.** The time that is left is inside the update
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

**That sentence now has a number on it, because a cheaper world was built and measured.**
`WalkerRagdoll-v0` takes a `physics_preset`: `v9` is the published one and changes nothing, `euler`
swaps RK4 for Euler at the same `timestep=0.002` so an action still covers the same simulated time,
and `fast` additionally drops self-collision while keeping every floor contact - the standing gate
counts feet-on-floor and penalises any other body on the floor, so those pairs are the ones the task
reads, and the 138-pair set it pays for is not. Measured in one process, back to back, 3 reps
(`benchmarks/physics_presets.json`):

| preset | what it changes | `env.step` n=1 | `env.step` n=8 | raw `mj_step` time |
|:---|:---|---:|---:|---:|
| `euler` | integrator only | 2.74x | 2.55x | 3.67x less |
| `fast` | integrator + no self-collision | 3.04x | 2.71x | 4.14x less |

Absolute rates in the same window were 3,668 env-step/s for `v9` at n=1 against 11,133 for `fast`,
with 1.5-3.2% rep spread - the ratios are the reusable part, as everywhere else in this section.

How far the cheap worlds drift from the published one is the other half of the measurement: 5
episodes of 1000 steps, identical resets and one shared `uniform(-1,1)` action sequence. `euler`
separates by up to 0.227 m of torso height and `fast` by 0.301 m, with mean observation L2 distance
28.18 and 28.42. What matters for the task's own criterion is that the gate's *inputs* survive:
over those 5,000 steps the reference holds 927 steps with a foot on the floor and 2,866 with some
other body on it, `euler` gets 830 and 2,961, `fast` 771 and 2,895. The step-by-step "did the contact
count change" counter says 3,691 and 3,678 - that figure measures *when* contacts happen, which two
integrators will never agree about, and it is reported here only so nobody reads it as a failure.
The number that decides whether any of this is worth using is the end-to-end one. Two identical
SAC screens - 1M steps, seed 7, `num_envs=8`, `target` phase, mixed resets - trained in each world:
**22 min 25 s in `v9`, 19 min 32 s in `fast`, which is 1.15x** where the environment step was
2.71x cheaper. That gap is the paragraph above, measured: the rest of the wall clock is the learner
and the pipes, not MuJoCo.

**The `num_envs=8` in those screens is not only a plumbing choice, and saying so costs a flag.** The
SAC loop runs one critic update per collection iteration, and a collection iteration is `num_envs`
environment steps (`train_walker.py:2099-2142`), so the optimisation each environment step receives
is 1/`num_envs`: at 8 envs a step is optimised four times as hard as at 32, and `--num-envs` was the
only knob that could turn that. The two rates in this section are the same fact wearing a stopwatch -
the `v9` arm of the 5M pair ran at **1,798.4 s per 1M steps** at 8 envs while the 40M run at 32 envs
sustained **467.4 s per 1M**, and 1798.4/467.4 is **3.85x**, which is the 4x update ratio and not an
engineering win. Collecting at 32 envs is cheaper precisely because it trains less per step. The
trainer now has the explicit knob the REDQ side already had (`--utd-ratio`, G updates per collection
iteration, default **1** - the shipped schedule, so no committed run moves).

**The arms that decide what it costs ran the same night** (`benchmarks/utd_dose_pair.json`): 5M at 32
envs, ratio 1 then ratio 4 back to back in one window - **2,399 s against 6,494 s, which is 2.707x and
not 4x**. Reading that pair as `C + U = 2,399` and `C + 4U = 6,494` (the updates one ratio-1 run
performs, plus the collection the two arms share) puts **1,365 s in the gradient step and 1,034 s in
collection: 56.9%** of a ratio-1 SAC run at 32 envs is the update. The Dreamer loop split reaches its
78.6% by instrumenting one iteration of a different trainer, so the two shares are two methods on two
trainers rather than one number confirming the other - what they agree on is only the direction, that
the update is the larger half. Buying the committed 8-env arm's dose at 32 envs cost **6,494 s where
that arm took 8,992 s (1.385x)**, and that is the one figure here carrying a window caveat: the two
arms were trained on different nights, and the same world at 8 envs ran at 1,798.4 s per 1M in one of
them and 1,877.2 s per 1M in the other. **The pair drawn a second time at seed 8 reproduces the
arithmetic** (`benchmarks/utd_dose_draws.json`): 2,529 s against 6,824 s, which is **2.698x**, 0.3% from
the first draw's 2.707x, and the same two equations put **56.6%** of that ratio-1 clock in the update.

The behavioural half is where the second draw changed the sentence rather than the number. With two
draws per dose, the higher dose reaches the target more often in **6 of the 8 cells that are not ties**,
and it does so at 5M in both seeds: **14 of 40 episodes against 4 of 40** (20% and 15% against 5% and
5%). Averaged over the three upper budgets it reaches in **19.17%** of episodes against **9.17%** for
the under-dosed arm. The mean return still ranks the arms the other way, on average
**32,157.9** against **29,266.1** across the two 5M draws - so after two draws per arm the return
column has *still* not moved with the task's own criterion, which is now measured on two different
axes: physics worlds and optimisation doses. What a higher dose buys is reaching, not score.

Two honest limits survive the replication. Two draws is still two: the 5M advantage is ten episodes out
of eighty, and the reach column inside each run moves by that much between neighbouring checkpoints.
And the whole comparison is one world and one algorithm - `v9`, SAC, `num_envs=32` - so the 2.698x and
the reach gap describe this trainer's schedule, not a law about doses.

And the caveat that limits what a screen can ask. Scored at the published protocol, the two arms
reach the target in 0 of 20 episodes each (`benchmarks/physics_presets_screen_v9.json`,
`..._fast.json`): mean 7616.16 against 4889.51, std 7572.40 against 16367.17, mean closest approach
3.266 m against 3.138 m. The 40M from-scratch curve in the Phase-1 section only begins reaching
between 5M and 30M, so a 1M-budget screen - cheap world or published one - cannot see the walking
question at all.

So the same pair was run again at 5M, the smallest budget where the behaviour exists
(`benchmarks/physics_presets_screen5m_paired.json`). Both worlds agree on the answer: each reaches
the target in 2 of 20 episodes at 5M, the published arm shows its first reach there and the `fast`
arm one checkpoint earlier (5.0% at 4M), and both stand at the end of an episode in 0% and 5% of
episodes respectively. The wall clock, measured back to back in that one window, is **2 h 29 min 52 s
in `v9` against 2 h 01 min 39 s in `fast` - 1.232x**, which is the same story the 1M pair told at
1.148x in its window: a 2.71x cheaper environment step is worth roughly a fifth to a quarter of a
SAC run's clock. The window matters more than it should: the `v9` arm of this pair ran at 1,798.4 s
per 1M where its sibling at 1M ran at 1,345 s per 1M, same world, same config, different night - so
only ratios measured inside a pair are reusable here, never absolute minutes.

What the pair cannot settle is level. The `fast` arm is further along at every budget (closest
approach 1.501 m against 2.309 m at 5M, falls 1.65 against 0.85 per episode), but there is one draw
per world and the two runs differ in world, not in seed, so nothing separates "the cheap world is
easier" from "this run got lucky". The screening conclusion is therefore about the *question*, not
the number: a screen in `fast` answers whether something reaches the target at all, and would not be
trusted to rank two knobs against each other. The two 1M windows further down are the evidence for
that distrust: the same recipe, scored twice, put a different world ahead each time.

The preset was then run on the trainer where physics should matter least, which is the case where the
loop split makes a falsifiable prediction. Dreamer, 250k env steps, 8 envs, seed 7, the captured CUDA
update in both arms, back to back in one window: **783 s in `v9` against 690 s in `fast`, 1.135x**
(`benchmarks/physics_presets_dreamer_pair.json`). The loop split says **78.6%** of that iteration is
the update and the preset makes `env.step` 2.707x cheaper at n=8, so leaving the update alone and
dividing only the collection part predicts **1.156x** - the run lands **1.8%** off the arithmetic.
That is the property worth having: a cheaper environment step buys the share of the wall clock the
environment actually occupies, so one loop split tells you in advance what a preset is worth on a
trainer you have not timed, and SAC (1.15x, 1.23x), PPO (1.156x) and Dreamer (1.135x) all came in on
the same side of that prediction.

`euler` had been measured on the step and on the drift and never trained in, which mattered because
it is the preset that stays closest to the published world. All three worlds were then run end to
end, one after another in a single window, SAC 1M at 8 envs and seed 7
(`benchmarks/physics_presets_sac1m_triple.json`): **3,044 s in `v9`, 2,630 s in `euler`, 1,960 s in
`fast`** - 1.157x and 1.553x. So the least-drifting world does buy speed, just not most of it, and
this window's `fast` gain (1.553x) is larger than the two earlier windows reported (1.148x, 1.232x),
which is the per-window spread the section keeps insisting on rather than a contradiction.

The column that decides what the presets may be used for is the last one, and it is not a speed
column. At the same budget the three arms do not tie: mean return **1,153.71** in `v9`, **5,861.46**
in `euler`, **17,690.50** in `fast`, with `fast` the only arm that ended any episode standing (5.0%
against 0.0%) and the only one that got meaningfully closer to the target (2.557 m against 3.057 m),
while also falling most (0.55 per episode against 0.10 - it is attempting). Unlike the
curriculum-versus-scratch comparison above, these three are measured under one reward function: all
three checkpoints record `target_forward_velocity` 1.2 and the same shaping, so the ordering *inside
this window* is a behavioural difference and not a bookkeeping artefact.

**Whether it is a difference between the worlds is already answered, and the answer is no.** The
first 1M screen (`benchmarks/physics_presets_screen_paired.json`) is this same recipe - SAC 1M, seed
7, 8 envs, target task, tv 1.2, cuda, 24 torch threads, each arm scored in the world it trained in -
and it found **7,616.16** in `v9` against **4,889.51** in `fast`, where this window found
**1,153.71** against **17,690.50**. The published world came down by a factor of 6.6, the cheap one
came up by 3.6, and the ordering reversed. The trainer re-seeds every run and makes no promise that
two runs of one config agree, so the honest reading is two draws of one recipe: at 1M the level column
is the run speaking, not the world. What did replicate is the thing a 1M screen is actually asked, and
it is a question rather than a number: **0 of 20** episodes reach the target in every arm of both
windows, in both worlds. A cheap world is therefore not a neutral faster lane - it leads or trails by
this much at the same budget, which is exactly what a screen is not supposed to introduce, and its
lead is not evidence at 1M. That is why the level question is deferred to a second draw at 5M with
both worlds re-run at a second seed rather than settled here.

Even the scorer feels the physics: the 20-episode evaluation of the three arms took 28 s, 21 s and
14 s respectively.

`eval_phase1.py` refuses to score a preset run against an aliased older revision, and a preset
checkpoint records its own world in `env_version`, so the two can never be mixed by accident.

**`euler` has now been trained at the budget where reaching is observable - twice - and the first answer
was wrong.** Draw 1 (seed 7) flat-lined: mean 3,150.93 falling to **-76.12** across its five checkpoints,
forward speed ~0, 0.10 falls per episode, the target reached in **0 of 100** scored episodes. Read on its
own that looks like a property of the integrator. Draw 2 (seed 8, `benchmarks/physics_presets_screen5m_euler_draws.json`)
is the same world at the same budget: it climbs to **14,409.67** at 4M and reaches the target in **10%**
of episodes at 5M. The flat line was the run, not the world - the inference the paired-seed design exists
to license, bought with 2 h 05 m of machine time instead of argued.

The cost half replicates on its own terms: **7,860 s against the published world's 8,992 s at seed 7
(1.144x)** and **7,505 s against 9,386 s at seed 8 (1.251x)**, two independent windows bracketing the
**1.157x** the 1M triple measured, each ratio computed inside its own draw. And the comparison this
section needed closes: `euler`'s two-draw reach record at 5M is **0% then 10%** where `v9`'s is **10% then
0%** - indistinguishable at this sample size - while `fast` led both of its draws. The world that drifts
least (0.227 m of torso height over a shared episode) tracks the published one on the task's own
criterion; the divergence lives in the world that prunes self-collision. Two draws is still two: 40
scored episodes per arm at 5M, and the two `euler` draws differ by more than 10,000 of mean return. What
is measured rather than assumed is that a 5M screen in `euler` asks the same question the published world
asks, at ~1.14-1.25x the speed.

**The second draw was bought for the level question, and it answered the two halves differently.**
The same 5M recipe - `num_envs=8`, target task, mixed resets, tv 1.2 - re-run with seed 8 instead of
7, both worlds back to back in one window (`benchmarks/physics_presets_screen5m_draws.json`). The
ordering replicated: the cheap world is ahead at **9 of the 10** world-by-budget cells, at 5M with
**62,311.88** against the published world's **-763.96**, reaching the target in 4 of 20 episodes
against 0 of 20. The one cell where `v9` leads (seed 7 at 2M, 15,328.48 against 14,087.03) is a
crossing on the way up, not a reversal.

The level did not replicate, and not by a small margin. The published world produced **18,956.17**
and 10% reach at one seed and **-763.96** and 0% at the other - a swing of **19,720.13** inside one
world, while the cheap one swung **23,532.97**. For scale, the distance between the two worlds in the
first draw was 19,822.74: the published world's own swing is as large as the gap it was being compared
across, and the cheap one's is larger still. So the number a single 5M draw prints is still the run
speaking; what two draws establish is that *which run is ahead* is the world. One of those figures
deserves its own sentence, because it is a result rather than noise: a 5M SAC run in the published
world, at a seed chosen before anyone saw its score, learned nothing that ever reached the target. The
Phase-1 curves in this repository are single draws of the same recipe.

The speed was the stable column all along: **1.232x** in the first window and **1.240x** in this one
(9,386 s against 7,568 s - 1,877.2 s per 1M against 1,513.6 s), and the scorer again felt the physics:
**78 s against 52 s** for the same hundred episodes.

### 🧮 The learner side: what was measured, what shipped, what was rejected

Collection is not the bottleneck for the Phase-1 algorithms, so "faster training" has to be
answered per trainer, at the same env-step budget. Every number below is wall clock on this
laptop (RTX 4070 Laptop 8 GB, 32 threads, `.venv`), one process at a time:

| Trainer | Config | Measured | 1M env steps |
|:---|:---|:---|:---|
| `train_ars.py` | linear policy, 10 directions | 1,005,153 steps in 544 s = 1848/s | 9 min |
| `train_walker.py --algo ppo` | 32 envs, `target` phase, seed 7 | 983,040 steps in 282 s = 3,486.0/s | 4.8 min |
| `train_walker.py --algo ppo`, 10M | same config, drawn at two seeds | 9,961,472 steps in 1,594 s = 6,249.4/s | 160.0 s |
| `train_walker.py --algo ppo`, 10M again | the other seed, in a window 2 h later | 9,961,472 steps in 3,636 s = 2,739.7/s | 365.0 s |
| `train_walker.py --algo sac` | same config, same night | 1,000,000 steps in 594 s = 1,683.5/s | 9.9 min |
| `train_dreamer.py` | 4 envs, update each collect step | 5,000 steps in 37 s with the update gate closed; 14,000 in 898 s in one A/B window and in 205 s later, same build | ~4-18 h |
| `train_dreamer.py`, captured update | same config, and the CUDA default now (`--no-update-graph` opts out) | 100 updates in 4.1 s against 17.6 s eager, in the same window | its own projection says 3.6 h; measured later at two budgets: **1.18 h** |
| `train_redq.py` | 16 envs, `utd_ratio=20`, ensemble 10 | 10,000 steps in 478 s | ~13 h |
| `train_redq.py`, batched ensemble | same config, `--ensemble-impl batched` (default now) | see the A/B below | ~4 h in that window |

**PPO is the fastest loop here and the worst policy at the same budget, and both halves of that
sentence are the measurement** (`benchmarks/trainer_pair_ppo_sac.json`, one night, both arms at 32
envs and `target`, seed 7). PPO collects 2.071× the environment steps per second that SAC does -
3,486.0/s against 1,683.5/s - because a rollout of 2048×32 steps buys 320 minibatch passes, roughly one
gradient step per 5 environment steps, while SAC updates about once per step. That is the whole of
the speed advantage and it is real. What it costs shows up in the score, not in the clock: at the
end of that budget PPO averages **-985.97** where SAC averages **22,020.06**, its closest approach to
the target is **3.211 m** against SAC's **2.325 m**, and neither reaches the radius in 20 episodes.
PPO's **falls per episode are 0.10 against SAC's 1.10**, which is not the good news it looks like:
`standing_at_end_pct` is 0 for both, so PPO is not falling because it is not attempting - it has
found the local optimum of lying still and collecting a small forward-velocity term.

One detail that has to travel with these numbers: PPO checkpoints only at rollout boundaries, so its
"1M" run ends at **983,040 steps**, not 1,000,000 - 1.7% less budget, which is why the rate column
above is quoted per collected step. The ordering is not an artefact of that one checkpoint: the same
night's earlier pair, `ppo_524288` against `sac_500000`, puts PPO at **3,150.48** mean and **3.318 m**
closest against SAC's **13,784.48** and **2.770 m** - same sign at 5% of budget and at 100%. Neither
arm reaches the target at either budget, and neither stands at the end of an episode at all.
The conclusion is about the question "faster" answers: switching trainer changes what the machine
can do with a budget, and at these budgets off-policy data is what moves the behaviour. That clause was
written when PPO had only ever been run here at 1M, and it has now been tested at ten times that
budget.

**The re-run happened, and the objection was the budget, not the algorithm.** Same config, 10M
requested, drawn at two seeds (`benchmarks/ppo_10m_draws.json`): the mean climbs from **-813.28** at the
first rollout boundary to **43,439.25** at 8,060,928 in draw 1, and from **4,397.74** to **34,707.81** in
draw 2; the best reach column is **15% (3 of 20)** at 5,046,272 in one draw and **25% (5 of 20)** at
7,012,352 in the other, with closest approach near **2.179 m** and **2.138 m** respectively. So "off-policy
is what moves the behaviour" was a sentence about 1M budgets: at 10x, the on-policy arm stands, walks and
reaches - in both draws.

**The clock did not replicate, and that was the claim the second draw was for.** The identical recipe
cost **1,594 s** in one window (9,961,472 steps at **6,249.4 env-steps/s**) and **3,636 s** in the next
(**2,739.7/s**) - a **2.28x** spread on one laptop, on AC power in both windows. That refutes the
sentence published from draw 1, that the whole PPO curve costs less clock than a SAC 5M arm in this
world: 3,636 s is *more* than SAC's ratio-1 arms (2,399 s and 2,529 s) and less than its ratio-4 arms
(6,494 s and 6,824 s). What survives is the weaker, honest version - PPO at 10M and SAC at 5M cost about
the same, straddling SAC's own dose choice - and the rate ratios quoted from draw 1 (**3.0x** against SAC
at ratio 1, **11.2x** against the 8-env screens) were cross-window numbers wearing a within-window look,
because no SAC arm ran in either PPO window. Why the two windows differ is not measured here: chain28 ran
two hours of GPU four minutes before chain29 started, so sustained throttling is the candidate and an
idle-gap replication is the test; both are recorded as open, and the rates are labelled per window so
neither can be quietly reused as a constant.

The behavioural limits are the familiar ones. Each draw is one run, and inside draw 1 the reach column
swings **0 to 15%** across ten checkpoints, draw 2's **0 to 25%** - the same band this section measured
when the published world moved from +18,956.17 to -763.96 between seeds. And PPO pays for its cheap
updates in falls: **2.55 to 3.2 per episode** at the top of draw 1 against the 1M arm's 0.10, which is
what attempting looks like. What the two draws do settle is the question the re-run was bought for:
PPO's deficit at 1M was budget. It took 27 minutes of machine time to learn that, and 61 more to learn
that its clock is not a constant either.

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

**The last lever in the loop, and what it costs.** With the update at 12.6-12.8 ms and the collection
side at 3.3-4.2 ms, the only way left to cut wall clock without touching the update's arithmetic is to
collect more environment steps per gradient step: the trainer runs exactly one `dreamer_update` per
collection step, so `--num-envs` divides the update's share directly. `python
bench_dreamer_update.py --mode amortize` measures that at two budgets per `n`, in one window
(`benchmarks/dreamer_amortize_n.json`):

| `--num-envs` | ms per iteration | env-steps/s | h per 1M env steps | gradient steps per 1k env steps | vs shipped |
|---:|---:|---:|---:|---:|---:|
| 4 (shipped) | 17.23 | 232.2 | 1.20 | 250.0 | 1.00x |
| 8 | 20.35 | 393.1 | 0.71 | 125.0 | **1.69x** |
| 16 | 28.38 | 563.8 | 0.49 | 62.5 | **2.43x** |

The n=4 row is the control, and it is why the other two are believable: 17.23 ms per iteration here
against the 16.96 ms the two-budget run measured and the 15.70 ms the in-process split attributed -
three instruments in three windows, 9% apart, and this one ran with 5 other CUDA contexts on the GPU
(the artifact records the count, because a first attempt at it ran against 6 and produced numbers
worth nothing). So a 1M-step Dreamer run at `--num-envs 16` costs **0.49 h** where the shipped
configuration costs 1.20 h. These are bench hours in one window with one collector backend; the same
ladder measured on three real runs is in "Does the cheaper update schedule still learn?" below, and
its absolute rates differ - the 16-env arm there also crosses `--vec-parallel-threshold` and runs the
process-parallel collector, so it went faster than this table predicts rather than slower.
The price is the last column: the update-to-data ratio falls from one
gradient step per 4 environment steps to one per 16, so 250 gradient steps per 1,000 environment steps
become 62.5. Whether that is a speedup or simply a shorter run that learns less is a learning question
and not a timing one, so it gets measured rather than assumed - see "Does the cheaper update schedule
still learn?" below.

**Does the cheaper update schedule still learn?** Six real runs: three update schedules × two seeds,
all to exactly 250,000 environment steps, all `--task-phase target --reset-mode mixed`, all scored
identically afterwards (50 seeded episodes, cuda, the training reward). The trainer runs one
`dreamer_update` per collection step and a collection step advances `num_envs` steps, so the arms spent
62,500, 31,250 and 15,625 gradient steps on the same experience. Wall clock is each run's own record -
the mtimes of the checkpoints it wrote, over the 50,000 → 250,000 window every run has, which excludes
startup and graph capture (`benchmarks/training_rate_history.json`):

| `--num-envs` | gradient steps | seed 7 | seed 8 | speedup within seed 7 | within seed 8 |
|---:|---:|---:|---:|---:|---:|
| 4 (shipped) | 62,500 | 1028.0 s, 194.6 /s | 930.0 s, 215.1 /s | 1.00x | 1.00x |
| 8 | 31,250 | 682.0 s, 293.3 /s | 583.0 s, 343.1 /s | **1.51x** | **1.60x** |
| 16 | 15,625 | 252.0 s, 793.7 /s | 425.0 s, 470.6 /s | **4.08x** | **2.19x** |

| at 250,000 env steps | 4 (shipped) | 4 | 8 | 8 | 16 | 16 |
|:---|---:|---:|---:|---:|---:|---:|
| seed | 7 | 8 | 7 | 8 | 7 | 8 |
| mean return | 6628.71 | 5323.76 | 8402.69 | 2790.46 | 5301.00 | 5643.10 |
| median return | 6466.67 | 5826.98 | 8320.57 | 2241.89 | 627.00 | 5840.20 |
| std | 4062.99 | 4746.02 | 4017.59 | 5380.37 | 7460.70 | 3795.10 |
| episodes inside the 0.45 m radius | 1 of 50 | 2 of 50 | 3 of 50 | 0 of 50 | 0 of 50 | 0 of 50 |
| mean closest approach | 2.647 m | 2.364 m | 2.456 m | 3.023 m | 3.176 m | 2.720 m |
| mean x-velocity | +0.0079 m/s | +0.0129 m/s | +0.0496 m/s | -0.0262 m/s | -0.0293 m/s | -0.0294 m/s |

**The first version of this section said the 8-env arm was free and the 16-env arm was a cliff. The
second seed refuted both halves, and the timing is what repeats.** Written from one seed each, the
8-env arm led by 1774 and the 16-env arm's median had collapsed to 627.00. Re-run at seed 8, the
8-env arm *trailed* by 2533 and the 16-env arm's median was 5840.20, its healthiest column. Each
schedule's own seed spread - 1304.95 across the 4-env mean and 5612.23 across the 8-env one - is
larger than any gap between schedules, and averaged over both seeds the means are 5976.24, 5596.58
and 5472.05: monotone in the right direction, and small enough that two seeds cannot rank them. So
what the return measures at this budget is the draw, not the schedule.

What does repeat is the **timing of the 8-env arm and the behaviour of the 16-env arm.** The 1.51x and
the 1.60x are 6% apart across seeds, so `--num-envs 8` is a real 1.5x. The 16-env arm's speedup is not:
4.08x at one seed and 2.19x at the other, which is why it is quoted as a range and why the isolated
bench's 2.43x sits inside it rather than being contradicted by it. And the column that does not wobble
is forward speed: the 16-env arm averaged **-0.0293 and -0.0294 m/s**, two runs agreeing to 0.3%, both
negative, and it reached the radius in **0 of 100 episodes**. The 4-env arm was positive in both seeds
(+0.0079, +0.0129) and reached it in 3 of 100. So the cost of the cheap schedule shows up as *not
moving toward the target* while the return - which this task pays mostly for posture, see the Phase-1
section - stays statistically the same. That is the cheapest example in this repository of why the
return is the wrong column to decide on.

Two confounds, both disclosed rather than argued away. At 16 environments the trainer crosses
`--vec-parallel-threshold` (default 16) and switches the collector from `sync` to process-parallel, so
the 16-env row is two changes at once; the 4- and 8-env arms both stay on `sync`, which is what makes
their 1.51x/1.60x a clean schedule comparison. And the windows differ: each run's starting GPU window
is recorded with it (255 MiB held by 4 other contexts for the 4-env seed-7 run, 626 by 5 for 8-env
seed 7, 531 by 5 for 16-env seed 7, and 618 by 5 for the 16-env seed-8 run, from MLflow params or, for
the one run that logged to a scratch database, its stdout). Each arm's `--num-envs` is corroborated
twice rather than taken on faith: the observation normalizer's counter exceeds `global_step` by exactly
that number, and the MLflow row repeats the flag the run was started with.

**So, for the question that started this - can training be faster?** Yes: `--num-envs 8` is 1.5x on
measured wall clock with nothing detectable lost over two seeds, and it is one flag. `--num-envs 16`
is 2.2-4.1x and, on the one column that replicated, it is the worst of the three at actually approaching
the target - so it is offered as a way to explore, not to train the result. Settling the ranking
properly needs three seeds per schedule at the 1M budget (about 4 h on this machine for the 8- and
16-env arms) and is **not measured**; the 1M comparison in the Phase-1 section is single-seed on both
sides. Changing the shipped default is deliberately not part of this either: `--num-envs 4` is the
configuration every other rate in this section was measured against, so moving it would invalidate
those numbers rather than improve them.

**Four of the six checkpoints here record their own reward.** The two that do not - the shipped arm's
250k checkpoint, which is the first quarter of `dreamer_v3_1m`, and the 16-env seed-7 run - were
trained before `train_dreamer.py` started saving `reward_kwargs` this evening, so they resolve through
the fallback and their rows say so. The other four report `reward_source = checkpoint`: scoring a run
trained from here on never has to reason about which trainer an algorithm name implies, which is the
whole point of the change.

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

**"It felt faster before all this work" - the archive says the opposite, by 16.5x.** Worth answering
with data rather than reassurance, because the intuition has a real source. `python
summarize_training_rate.py` (`benchmarks/training_rate_history.json`) recomputes a rate for every run
in `mlruns.db` that logged `sps` twice or more - the artifact lists them all, and the list only grows -
as (last step − first step) / (last timestamp − first timestamp), which needs no interpretation of
what the trainer meant by `sps`. The
two it cites are the furthest each generation of the Dreamer trainer got, and they are matched on
every parameter that is recorded (`algo=dreamer`, `num_envs=4`, `task_phase=target`, `seed=7`,
`total_timesteps=1000000`, `reset_mode=mixed`); the only difference between them is the code:

| run | generation | started | steps measured | span | rate | h per 1M steps | first logged `sps` | last logged `sps` |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|
| `dreamer_dreamer_v2_1m_v9__7` | before capture | 2026-10-02 11:30 | 263,000 | 19,981.7 s | 13.2 /s | 21.10 | **235** | 13 |
| `dreamer_dreamer_v3_1m__7` | after capture | 2026-10-04 15:01 | 995,000 | 4,565.4 s | 217.9 /s | **1.27** | **690** | 218 |

**16.51x**, and it is not two lucky draws: all 11 Dreamer runs in the archive from before the capture
landed sit between **9.3 and 23.7 env-steps/s**. The 4,565.4 s span is also an independent
confirmation of the 1.27 h this README quotes for the completed run - MLflow's metric timestamps
against the trainer's own clock, agreeing to the minute.

The intuition's source is in the last two columns. `sps` is
`int(global_step / (time.time() - start_time))` - a **cumulative average since process start**, and
its first sample is logged at the first step the run actually reaches at or after `learning_starts`
(5000 for a 4- or 8-env run, 5008 for a 16-env one, because `global_step` advances `num_envs` at a
time) - before a single gradient update has run, so it
reports the collection-only rate and then decays for the rest of the run. Opened in the first minutes,
the old trainer's chart read **235** and the new one reads **690**; left alone, they settle at **13**
and **218**. The old runs were also the ones that died early - that one has a 1,000,000-step budget
and its last checkpoint is at 269,404 - so the decay never landed on screen. A rate read off the top
of that curve is the collection rate, and it was higher before too - it just was not the run's cost.

Two limits on what this can claim. The archive's `sps` metric only exists from 2026-10-01 onward; the
21 earlier runs span 2026-06-07 to 2026-06-26 and are offline RL (BC/IQL/CQL) with no rate metric at
all, so there is no measured "a week ago" to compare against - the nearest earlier generation of the
same trainer is the 2026-10-01/10-02 one above. And env-steps/s is not comparable across algorithms:
the stacked SAC harness measures **2,883-4,049 env-steps/s** because SAC spends one small-MLP gradient
step per environment step, where Dreamer spends a world-model update per four. If the memory is of
watching a SAC run, that is the difference - what a step costs, not how fast the machine was.

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

The probe was run again a week later in a fresh WSL2 venv, and it moved the answer the wrong way for
a port. Same script, same GPU, jax 0.11.2 where the numbers above came from 0.10.2: the rollout
costs **1.741 ms** forward-only and **8.354 ms** with `value_and_grad` - the gradient path is now
**2.6x** the captured torch loop it would replace, against 1.6x in the first probe. The GPU was not
more busy: the fp32 matmul floor measured *faster*, **19.5 TFLOP/s** against 15.2, putting the
arithmetic floor at **0.01 ms** and 0.58% of the loop. A device with more headroom and a loop with
less speed is what dispatch-bound looks like from the outside, and it is the second time this
particular port has measured against itself (`benchmarks/jax_rssm_imagination_wsl_jax0112.json`,
committed beside the original rather than over it - a jax version is not the same instrument).
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
harness — **369 tests, 219 s in this window** (`Ran 369 tests in 218.984s ... OK
(skipped=7)` under `.venv`). Windows of this suite have measured 176.3 s at 102 tests, 269.995 s
at 121, 261.1 s at 127, 329.964 s at 128, 319.168 s at 130, 184.716 s, 203.108 s and 306.976 s at
140, 144.678 s at 147, 230.268 s at 157, 171.016 s and 170.304 s at 194, and 174.008 s,
170.391 s, 175.036 s, 168.775 s, 168.986 s and 166.606 s at 201/205/210, 169.919 s at 214,
and 163.920 s and 161.829 s at 237, 165.360 s and 164.902 s at 243, 161.647 s and 162.674 s at 250, 162.726 s and 162.167 s at 251, 171.616 s and 163.434 s at 252, 163.769 s and 163.748 s at 255, 226.912 s and 217.651 s at 260, 193.167 s and 191.909 s at 273, 206.150 s at 279, 193.904 s at 284, 201.464 s at 289, 189.864 s and 187.485 s at 294, 190.012 s at 297, 230.180 s at 301, 190.655 s at 311, 211.202 s at 317, 195.419 s at 322, 184.182 s at 326, 180.964 s at 326, 266.112 s and 191.187 s at 339, 185.556 s at 340, 193.319 s at 351 -
those last windows carry a dose test that runs three short CPU trainings, which are about 27 s of
them, so that entry is not slower hardware; consecutive runs of one commit agree to 4%, where the
147 and 157 windows an afternoon earlier were 1.6x apart for ten more tests. The
duration belongs to the machine's state, the
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

The history lives in **one** database at the repository root: **154 active runs** in 6 experiments
(133 `walker-ragdoll`, 10 `Walker_Offline_To_Online`, 8 `Walker2d_Offline_to_Online`,
2 `Walker_Behavioral_Cloning`, 1 `Walker_OpenAI_BC`, 0 `Default`) and **284,919 metric rows** as of
2026-10-04 — and 103 of them carry an `integration_test`/`smoke`/`bench` name, because the training
tests wrote here until they were pointed at a throwaway backend (`tests/test_training.py` now sets
`MLFLOW_TRACKING_URI`, and the benches always did). Read the run count as an archive of everything
this repository has ever executed, not as 154 research runs - and read it as a floor, because every
training run adds to it and the gate only checks that the published figure is not above the database.
`utils/mlflow_uri.py` resolves the path
from the repo root, so the working directory no longer decides where a run goes. 13 runs killed
mid-training were closed with a
`closed_as_stale` tag rather than deleted, because the metric history is the only surviving evidence
that they ran - see Phase 4, item 6. How each of those states was found, with the commands, is in
[docs/lab-notes.md](docs/lab-notes.md).

## 🔬 Reproducing and measuring

```bash
python -m unittest discover -s tests -t .   # 369 tests in .venv, 219 s; see "Running the tests"
python bench_env.py --seconds 4             # env throughput, physics vs Python split
python bench_mjx.py --sizes 32,128          # MJX/JAX batched stepping
python verify.py                            # Phase-2 artifact check (exits 2 when missing)
python bench_device.py                        # SAC update cost, CPU vs CUDA
python bench_dreamer_update.py --mode scaling # per-loop-step cost, eager vs captured, 3 batch points
python bench_dreamer_update.py --mode loop-split  # one iteration, segment by segment, one window
python bench_dreamer_update.py --mode real-rate   # the shipped command at two budgets, 2 reps/cell
python bench_dreamer_update.py --mode checkpoint-cost  # bytes and ms per Dreamer save, per interval
python bench_dreamer_update.py --mode amortize  # rate and gradient steps per --num-envs, one window
python bench_posture.py --episodes 50           # does it stand? torso height per step, not the return
python bench_posture.py --compare-devices       # the same rows on cuda and cpu, and what moves
python bench_physics_presets.py --skip-divergence  # what each physics_preset costs, back to back
python bench_physics_presets.py --skip-throughput  # how far each one drifts from v9, same actions
python bench_reach_definitions.py             # arrival by distance vs arrival standing, both rules,
                                              # and it refuses to write unless the loose column
                                              # reproduces each committed artifact episode for episode
python bench_approach_mechanism.py            # the same episodes step by step: metres of approach
                                              # closed inside the standing band vs below it, longest
                                              # continuous stand, velocity toward the target while up
python bench_mpc.py --episodes 20 --samples 24 --iterations 2 --elite 6 --horizon 60 --replan 10 \
  --out benchmarks/mpc_target_baseline.json   # the sampling-MPC baseline, same episodes and same
                                              # per-step instrument; --samples 48 --iterations 3
                                              # --elite 12 --horizon 100 --replan 20 is the
                                              # bigger-search cell
python collect_mpc_demos.py --episodes 28 --samples 48 --iterations 3 --elite 12 --horizon 100 \
  --replan 20                               # the teacher's own trajectories, on seeds disjoint from the
                                              # scored protocol (101+); arrays are gitignored, the
                                              # sidecar .json with their provenance is not
python train_bc_ragdoll.py --filter raw     # BC student on the SACAgent architecture; --filter band
                                              # trains only on the teacher's in-band steps
python bench_bc_teacher.py --jitter-compare benchmarks/demos/probe_elite.npz \
                                          # is the teacher's action learnable at all? constant vs ridge
                                          # vs k-NN vs each student, on one episode-level split
python collect_mpc_demos.py --episodes 28 --samples 48 --iterations 3 --elite 12 --horizon 100 \
  --replan 20 --demo-seed-base 401 --out benchmarks/demos/mpc_preference_demos.npz
                                          # the same teacher, now also recording its lookahead value per
                                          # step, on seeds 401-428
python train_offline_critic.py --actor q  # twin critic fitted by TD on those transitions and an actor
                                          # ascended through it; --actor q_bc adds TD3+BC's
                                          # in-distribution term. 60 epochs, 3 episodes held out, and
                                          # the checkpoint is the best held-out TD error, not the last
python eval_phase1.py --num-episodes 20 --seed 11 --reset-mode mixed \
  --model critic_q=checkpoints/critic_mpc_q/critic_student_best5.pt \
  --model critic_q_bc=checkpoints/critic_mpc_q_bc/critic_student_best5.pt \
  --out benchmarks/critic_mpc_students.json   # both students on the published protocol
python bench_bc_teacher.py --demos benchmarks/demos/mpc_preference_demos.npz \
  --students benchmarks/critic_mpc_students.json \
  --out benchmarks/critic_teacher_benchmark.json
                                          # ridge R2 against the recorded value, and each student's
                                          # action distance against a constant zero
python train_walker.py --algo sac --run-id preset_sac_euler_1m --seed 7 --total-timesteps 1000000 \
  --num-envs 8 --task-phase target --reset-mode mixed --target-forward-velocity 1.2 \
  --checkpoint-interval 500000 --device cuda --physics-preset euler
                                              # one arm of the 1M triple end to end; `v9` is the
                                              # same command with the last flag dropped
python eval_phase1.py --num-episodes 20 --seed 11 --reset-mode mixed --physics-preset euler \
  --model sac1m_euler=checkpoints/preset_sac_euler_1m/sac_actor_1000000.pt \
  --out benchmarks/physics_presets_sac1m_euler.json   # its score, in the world it trained in
python train_walker.py --algo sac --run-id utd_sac_n32_r4_5m --seed 7 --total-timesteps 5000000 \
  --num-envs 32 --utd-ratio 4 --task-phase target --reset-mode mixed \
  --target-forward-velocity 1.2 --checkpoint-interval 1000000 --device cuda
                                              # --utd-ratio is the optimisation dose: G gradient
                                              # updates per collection iteration. 1 is the shipped
                                              # schedule every committed run used
python train_walker.py --algo sac --run-id stab60_sac_n32_r4_5m --seed 7 --total-timesteps 5000000 \
  --num-envs 32 --utd-ratio 4 --task-phase target --reset-mode mixed \
  --target-forward-velocity 1.2 --checkpoint-interval 1000000 --device cuda \
  --no-save-replay-buffer --reward-override stability_reward_weight=60
                                              # --reward-override replaces one shipped reward weight for
                                              # one run; the effective dict is what the checkpoint records,
                                              # and task parameters (radius, distance range, horizon) are
                                              # refused on purpose
python train_walker.py --algo sac --run-id tcur_sac_n32_r4_5m --seed 7 --total-timesteps 5000000 \
  --num-envs 32 --utd-ratio 4 --task-phase target --reset-mode mixed \
  --target-forward-velocity 1.2 --checkpoint-interval 1000000 --device cuda \
  --no-save-replay-buffer --target-curriculum
                                              # targets start at 2.0-2.5 m and open 0.5 m every three
                                              # successes; the checkpoint stamps _tcur, and
                                              # `eval_phase1 --target-curriculum off` asks the transfer
                                              # question on the published 2-5 m
python summarize_training_rate.py               # every run's real rate, from its own logged timestamps
python summarize_curriculum_provenance.py       # the manual curriculum's order, widths and stage change
python -m utils.gpu_window                    # the GPU window a long run would start into
python bench_jax_update.py                    # JAX imagination rollout, fwd + fwd/rev (needs a
                                              # CUDA jaxlib: WSL2, see the MJX section). The 0.11.2
                                              # re-run is in a venv there, not over the committed
                                              # 0.10.2 capture:
                                              #   wsl: XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 \
                                              #     ~/.venvs/jaxlab/bin/python bench_jax_update.py \
                                              #     --out benchmarks/jax_rssm_imagination_wsl_jax0112.json
```

`verify.py` currently exits non-zero because `dataset.csv` is not in the repository — see the
Phase-2 caveat. That is the accurate answer, not a defect in the checker.
