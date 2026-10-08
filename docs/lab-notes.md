# Lab notes

Operational history of this machine and this repository: how measurements got lost, how the
environment got repaired, what was deleted and what it cost. The README carries the rules and the
numbers; this file carries the diary behind them, so the README can stay about the experiments.

Nothing here is a result. If a number appears, it appears because it was the evidence for a
decision, and it keeps the artifact path it came from.

## Losing a long run to a shared GPU

The Dreamer 1M attempt of 2026-10-02 got to 84,456 steps and died with `RuntimeError: CUDA error:
unspecified launch failure` - inside a window in which two other processes were also using the
CUDA device (a Phase-3 re-score and a checkpoint sweep). It left no checkpoint, because
`train_dreamer.py` wrote one at the default 200k interval and 84k steps had not reached it. The
queue was relaunched detached with `--checkpoint-interval 100000`; the shipped default has since
gone to 50000 on measured cost (README, "What was done about it"), and `utils/gpu_window.py` refuses
a long run into a contended window.

The same sharing is why absolute rates in this repository are quoted with their window. Two runs of
one identical Dreamer build measured 15.6 and 68 env-steps/s, and five desktop processes (Medal,
Overwolf, two Brave renderers, one the driver will not name) hold a CUDA context at all times,
which is the idle state - not the contended one.

## Rates read off training logs are wrong

The trainers print only when an episode ends, so sampling the last `global_step=` line twice a
minute reported 48 env-steps/s for a Dreamer that actually runs at 15.6. Only wall clock over a
whole run is trustworthy.

A second way the same log misleads, found 2026-10-04: the `sps` metric is
`int(global_step / (time.time() - start_time))`, a cumulative average since process start, and its
first sample lands at `learning_starts` before any update has run. Every Dreamer run in the archive
therefore opens with a sample that reads like the collection rate - 235, 423, 683, 782, 826 - and
decays from there to what the run costs, 9.3 to 23.7 env-steps/s before the CUDA graph and 218 after.
The old runs were also the ones that died early, so the decay never appeared on screen and the top of
the curve was the last thing anyone saw. `python summarize_training_rate.py`
(`benchmarks/training_rate_history.json`) recomputes each run's rate from the metric's own timestamps
instead of reading its values, which is immune to both traps.

## A claim about an algorithm, measured at one point on the budget axis

The learner-side section said "for this task off-policy data is what moves the behaviour", and the
evidence behind it was real: one night, both arms at 32 envs and ~1M steps, PPO averaging **-985.97**
with 0 of 20 reaching and a closest approach of 3.211 m while SAC averaged **22,020.06**. What the
sentence did not say is that it had been tested at exactly one budget, and PPO's own weakness at small
budgets is the textbook reason not to. The re-run at 10M (chain25, 1,594 s of wall clock) closes the
gap: the same recipe climbs from **-813.28** at its first rollout boundary to **43,439.25** at
8,060,928 steps, reaches the target in **15% of episodes (3 of 20)** at 5,046,272, and ends episodes
standing 5-10% of the time where the 1M arm ended standing 0%. On-policy learning at this task does
move the behaviour - late, and after ten times the environment steps.

**The general form: when a conclusion names an algorithm class, check whether the evidence names a
budget.** The correction cost 27 minutes and one sentence; it was free to spot beforehand, because the
pair's own artifact already recorded that 1M was PPO's first checkpoint-reachable boundary. A related
trap surfaced while writing it up: the new run is **3.0x** the rate of the SAC arm measured over the
same 5M budget and **3.7x** the rate of the SAC arm from the committed 1M pair, and both numbers are
correct. The pair's SAC rate (1,683.5/s) divides fixed startup by 1M steps while the 5M arms divide it
by five times as many, so the artifact stores both ratios and says which one to lead with. Same shape
as the short-run projections this file already documents, arriving on the other side of a comparison.

The PPO arm is one draw, and its reach column swings 0-15% across its own ten checkpoints - the same
band the seeded-sibling experiment measured the previous evening in SAC. So the published claim is the
shape (PPO comes up late and reaches occasionally) and the cost (the whole curve is cheaper than one
SAC 5M arm), not a ranking of the two algorithms at any budget.

## A score without its device is half a measurement

`eval_phase1.py` wrote the protocol, the env revision and the reward source into its artifact but not
the device it ran on, so nothing downstream could tell that five published rows had been scored on
cpu while the rows they were compared against were scored on cuda. One of them carried a conclusion:
"more training did not buy a better policy" compared 6486.27 at 269,404 steps against 5215.46 at
1,000,000, and the first of those was a cpu number
(`benchmarks/phase1_dreamer_v9_269k_cpu_reproduction.json` reproduces it exactly). The comparison
survived the correction - both rows on cuda are 6368.15 and 5315.86 - but it had not been the
comparison it claimed to be.

The size of the effect is what makes this a rule rather than a nitpick: same checkpoint, same seeds,
same reward, 20.6% apart for a stochastic Dreamer policy and 30% apart for a *deterministic* SAC
actor (23589.07 cpu against 17363.56 cuda, `benchmarks/phase1_evidence_eval_cpu_auto.json` against
`benchmarks/phase1_evidence_eval.json`). Seven of those ten episodes are on different trajectories.
The explanation this repository published first - that only policies with a sampled latent move -
was wrong for exactly that reason; the mechanism is a chaotic thousand-step rollout amplifying
float32 rounding, which is why ARS (one linear map in float64 numpy) and the zero-action reference
are the two entries that do not move at all. `eval_phase1.py` and `bench_posture.py` now both record
`device` and `torch_threads`.

## A roster key is not a label

`bench_posture.py` held one entry keyed `sac_40m` pointing at `sac_ckpt_9000000.pt`. The table
printed, the row read plausibly, and the README described a 9M policy as the 40M one for a day - the
row that the whole standing-band argument ranked first. Both checkpoints are now separate rows
(14.73% and 10.39% of steps inside the band), every row records the path it was scored from, and a
gate compares the step count in the label against the step count in the filename. The general form:
a harness that takes its row labels from a dict key will eventually be handed a key that lies, and
only the recorded path can catch it.

## A long job started with `nohup` dies with the call that started it

Three runs died in one afternoon, silently, with no traceback and the wrapper's exit line never
written: a detached-looking `nohup ... &` started inside a shell call is killed with that call's
process tree some minutes later. Launch long jobs detached through the harness that tracks them,
not through a backgrounding operator in a command that returns.

## The mlflow 3.x wall, and the error that was swallowed

`mlruns.db` is at schema revision `b7e2c1a4d9f3`, which only mlflow 3.x understands. With the
`mlflow 2.17.2` that the old Python 3.8 `.venv` shipped, opening it raises
`alembic.util.exc.CommandError: Can't locate revision identified by 'b7e2c1a4d9f3'` - and because
the trainers' mlflow helper is deliberately fault-tolerant, that error was swallowed, so runs
appeared to log while writing nothing. `requirements.txt` pins `mlflow>=3.0` for that reason. Found
while adding the stale-run tool, which had to open the same database.

mlflow 3.x publishes no Python 3.8 distribution - `pip download --no-deps "mlflow>=3.0"` returns
`No matching distribution found` - so the fix had to be the interpreter, not a `pip install`. `.venv`
is now Python 3.11.9 with mlflow 3.16.1, and the logging path was checked end-to-end:
`start_mlflow_run` -> `log_mlflow_metrics` -> `end_mlflow_run` against a throwaway sqlite backend
produced a FINISHED run carrying both metrics and the `seed` param. The swap is numerically
invisible: the same 200-step rollout from seed 7 gives reward -2506.5429333387096 in the 3.8 and
3.11 environments, and `evaluate_merging.py --num-episodes 2 --seed 11` reproduces all four strategy
means to the last printed digit. The old environment is kept as `.venv-py38-backup/` (two `mv`
commands to go back), and `start_mlflow_run` still prints the version and the remedy when it hits
the wall, so a freshly-cloned 3.8 venv fails loudly instead of silently.

## One database instead of two

The history now lives in one database, at the repository root: 21 runs, 5 experiments, 252,856
metric rows. It used to be two: 11 Phase-4 scripts wrote `sqlite:///mlruns.db` and 5 wrote
`sqlite:///../mlruns.db`, and because `run_all.bat` runs from inside `openai_walker/`, half the runs
landed in `openai_walker/mlruns.db` while `mlflow ui` from the root showed only the other half.
`utils/mlflow_uri.py` (wrapped by `openai_walker/mlflow_backend.py`) now resolves the path from the
repository root, so the working directory no longer decides where a run goes. The 13 runs that had
already landed in the wrong file were merged in with
`python openai_walker/mlflow_backend.py merge --source sqlite:///.../openai_walker/mlruns.db --apply`;
the source file is kept untouched as `openai_walker/mlruns.merged-into-root.db`, and the merge is
keyed on `run_uuid`, so re-running it moves nothing (verified: `0 new run(s)`).

Three runs were left in RUNNING by killed processes - `AIRL_IRL` (74,297 metric rows),
`Deep_PQR_IRL` (0 rows) and `IQL_SAC_FineTuning_Walker2d` (653 rows), started 97, 96 and 115 days
before that was written. `python openai_walker/mlflow_backend.py stale-runs --apply` closed them and
stamped a `closed_as_stale` tag on each, because `set_terminated` writes today's date into
`end_time` and a run that died in June would otherwise read as an 87-day training session. Every
trainer already wraps its work in `with mlflow.start_run(...)`, so these are abandoned-by-kill runs,
not missing `end_run` calls - and they are the reason an artifact can be absent while its metrics
are present.

## The unreachable 1.15 GB blob

`.git` was 1.2 GB while the real history is only ~28 MB. The difference was a single unreachable
object: `openai_walker/extratrees_model.pkl` had been `git add`ed (force-added past the `*.pkl`
ignore rule) on 2026-06-28 and later reset, leaving the blob dangling - no commit ever referenced
it, so `git rev-list --objects --all` did not show it and GitHub never received it.
`git prune --expire=now && git gc --prune=now` removed it and `git fsck` is clean; the history is
untouched and the clone is small again.

## The day `import sqlalchemy` stopped working in `.venv-phase4`

The GAIL retrain of 2026-10-05 died after seven seconds: `ImportError: DLL load failed while
importing _processors_cy: uma política de Controle de Aplicativo bloqueou este arquivo`. The
diagnosis was slow because `import mlflow` succeeds in that venv - mlflow does not pull sqlalchemy
at import time - so the failure only appears at the first `set_uri()`, which is the line every
Phase-4 script shares. The whole phase is one import away from working.

What changed was not this repository. `sqlalchemy` 2.1.1 landed in `.venv-phase4` on 2026-10-01
(`sqlalchemy-2.1.1.dist-info` mtime 10:59). 2.1.x imports its Cython modules at module scope with no
guard (`util/_has_cython.py` calls `_all_cython_modules()` unconditionally), and Windows App Control
blocks those unsigned `.pyd` files in this venv - the same policy that blocked matplotlib's
`_c_internal_utils` in a fresh venv here. 2.0.54 wraps the identical import in
`try/except ImportError` and falls back to `_py_processors`, so it runs with
`HAS_CYEXTENSION == False` and a message naming the blocked file. `pip install "sqlalchemy<2.1"`
therefore fixes the phase; `requirements-phase4.txt` now pins it, with the reason next to the pin.

**Copying the working bytes did not work, and that is the part worth remembering.** The
`_processors_cy.cp311-win_amd64.pyd` in `.venv` is hash-identical to the one in `.venv-phase4`
(both `d376a410...6d5e36`), and the former loads while the latter is refused. Replacing the blocked
file with a copy of the allowed one changed nothing - the decision is not made on file content
alone, so "it works over there" is no argument for "copy it here". The supported route is the
version that has a pure-Python fallback.

## App Control escalated mid-session, twice, to the trainer stacks themselves

On the night of 2026-10-05/06 the same policy that had blocked `sqlalchemy` 2.1.1's Cython modules in
`.venv-phase4` started blocking two more files, and the timing was in the middle of running work:
`.venv`'s `torch/lib/torch_python.dll` (16 MB, installed 2026-10-01 17:57) began refusing to load
with "uma política de Controle de Aplicativo bloqueou este arquivo" at ~02:25, twenty minutes after
the same interpreter had finished a green 251-test run; and `.venv-phase4`'s `mujoco/_functions`
began the same, after that venv had stepped MuJoCo physics for a 5M-step SAC run at 01:36-01:51.

**The blocks are per file and they are complementary, which is what makes the box unable to run
anything end to end.** Measured three times in a row over several minutes, they did not waver:
`.venv` = numpy ok, mujoco ok, torch blocked; `.venv-phase4` = numpy ok, torch ok (CUDA visible),
mujoco blocked; `.venv-mjx` = mujoco ok, no torch. No single Windows interpreter on this machine had
both a working tensor library and a working physics library, which is exactly the pair every
Phase-1 trainer needs.

**About fifty minutes later both lifted by themselves** - same files, same paths, nothing reinstalled
and no policy edited here, and the suite then ran green at 252 tests. So the accurate description is
not "a file got blacklisted" but "the verdict is re-evaluated at load time and can say no for a
while", and that changes the right first response: when a compiled extension of a venv that has been
working all night suddenly refuses to load, wait and retry before touching the environment.
Reinstalling would have replaced files that were about to work again, and the pinned pair
(`torch 2.4.1+cu121`, `mujoco 3.2.3`) is what the golden-reward tests are calibrated against.

Nothing published moved: the runs that produced the numbers of that night completed before the blocks
landed, and their artifacts are committed. What stopped is the ability to *verify* a new number -
`unittest discover` cannot import the suite, so a README claim added after 02:25 cannot be gated
against a measured run, and that is the rule this repository is built on. The JAX re-measurement
paragraph waited for the retry above rather than going in on the strength of the numbers alone.

Two escape routes, both of which change the instrument rather than fixing it. WSL2 has no such
policy: it ran `jax 0.11.2` on a `CudaDevice` in minutes (`nvidia-smi` works inside it, the driver is
shared), which is why the learner probe could be re-measured at all while Windows could not run the
suite - but a Linux run of the test suite is a different platform for the tests that pin rewards to
1e-9, so it cannot certify the Windows numbers. GitHub CI is the other one: it runs the same tree on
Linux with its own environment, and it prints the `Ran N tests in X s` line the README quotes, at a
duration belonging to a runner rather than to this laptop.

Reinstalling the blocked packages is the obvious third move and probably not the right one: the
previous incident showed the verdict is not made on file content alone (a byte-identical copy of a
`.pyd` that loads in one venv was refused in another), so re-downloading the same wheel rebuilds the
same hashes, and installing a *different* version to get different bytes is what broke `sqlalchemy`
in the first place and would here move the pinned `torch 2.4.1+cu121` / `mujoco 3.2.3` pair that the
golden-reward tests are calibrated against.

## Retraining a policy whose checkpoint the README measures

The 2026-10-05 GAIL retrain wrote over the file the published Phase-4 table was measured on:
`train_irl_gail.py` saves to `gail_model.pt`, and that exact name is what `evaluate_all.py` scores.
It ran inside a wrapper that hashed the three files at risk (`gail_model.pt`,
`final_results_50ep_seed2026.txt`, `final_episodes_50ep_seed2026.json`), copied them aside, trained,
scored the fresh weights, restored the June weights and scored them again as the control, then put
all three back and re-hashed. All three came back identical
(`01380edb…`, `6b4a53d5…`, `e52a217d…`), the fresh policy is kept under its own name as
`gail_model_retrain_2026-10-05.pt`, and the control reproduced the published row to the last digit -
which is the check that the wrap worked rather than a claim about the policy.

**The reports are the fragile half.** `evaluate_all.py` names its outputs after the protocol only -
episodes and seed - so scoring one model at 50 episodes and seed 2026 replaces the tracked
thirteen-model capture with a one-model file. Nothing in the repository warns about that, and the
ignore rules do not cover it either, because those files are tracked. A single-model capture at a
published protocol is therefore only safe with the copy-aside step, and the two GAIL captures this
session produced are committed under names that say which policy each scored.

The trainer also gained a checkpoint every 100k steps during this work. It had exactly one write,
after the final step, which is the pattern that already cost this repository an 84k-step Dreamer run
(above); the retrain was three hours and eighteen minutes, and losing it at 950k would have been the
same accident.

## The eighteen deleted recovery checkpoints

Everything under `checkpoints/` is local and ignored, so none of it ever affected the clone size.
The working directory here was 31 GB, of which 12.8 GB was `checkpoints/walker_recovery_v1`: twenty
hourly checkpoints from the same 20M-step run. Two carry the published Phase-3 numbers
(`sac_ckpt_20000000.pt`, the recovery expert) and the transfer-learning starting point
(`sac_ckpt_1000000.pt`, `merge_models.py --base-ckpt`); the other eighteen were deleted on
2026-10-01, freeing 11.4 GB (`checkpoints/`: 12,853 MB -> 1,454 MB), and the Phase-3 evaluation
reproduces its four strategy means exactly afterwards (36496.42 / 31237.52 / -8015.85 / -30245.87 at
two episodes, seed 11).

**"No script names them" is not the same statement as "nothing can use them", and the difference
matters.** Any checkpoint in a run directory is addressable by step number:
`play.py --run-id <id> --checkpoint-step <step>`, `--checkpoint <path>`,
`merge_models.py --base-ckpt/--rec-ckpt`, and
`train_walker.py --init-from-run-id <id> --init-from-checkpoint-step <step>`
(`resolve_checkpoint_any`, which is how the recovery run was seeded from `walker_target_v1` in the
first place). Those eighteen were therefore valid inputs for a linear-mode-connectivity or
task-vector sweep *along the recovery trajectory* - interpolate 2M against 19M, or ask when the
recovery policy left the walking policy's basin - and that study now needs the 20M-step run repeated
to do. Nothing published here depended on them, but they were not dead weight, and this was the more
irreversible of the two available choices. The same sweep remains open on the other expert
(`walker_target_v1` keeps 41 steps, each as `sac_actor_*` plus `sac_ckpt_*`), and the Phase-1 runs
recorded in the README checkpoint every 200k steps so the analysis is possible on them from the
start.

## Asking the process table whether the machine is free is not a lock

Two chains trained on this laptop at the same time on 2026-10-06, and two arms had to be thrown out.
The launcher logs are the record: `chain22_evidence.log` has pid 37556 writing
`12:52:36 machine clear, proceeding / START SAC 1M in v9`, and `chain23.log` has pid 54604 writing
`12:52:36 machine clear, proceeding / START second draw SAC 5M in v9`. Same second, because both were
launched at 12:51:35 and both pre-flight checks are the same loop - "sample the process table for a
`train_*.py`/`eval_*`/`bench_*`/`pytest` command line, require two consecutive clear readings 30 s
apart". At 12:52:36 neither chain had started its interpreter yet, so each one correctly reported an
idle machine and then went on to occupy it. The scan cannot see a competitor that has not started, so
two chains launched together do not race each other for the lock - they both pass, always.

About three minutes of two SAC trainers on one box is what the two arms got before they were stopped,
and both were discarded rather than repaired: the `v9` window published in
`benchmarks/physics_presets_sac1m_triple.json` is `12:55:56 -> 13:46:40`, the restarted copy's, and
the 12:52:36 arm exists only as two orphan log lines. Nothing in either arm's output shows the
collision - both would have exited 0 with plausible numbers, at ratios that would have been wrong.

**What licenses the published arm is its stdout, and not the argument I first believed.** Both copies
of a chain redirect the trainer to the same path (`%TEMP%\c22_v9.txt`), and I read that as the second
`Start-Process` being unable to open the file until the first interpreter exited. It is not: tested
afterwards, Windows lets a second redirect open a file another process is still writing, so the
redirect proves nothing about exclusivity. What the file shows is its own contents - one monotone
stream from `global_step=176` to `global_step=999400`, no restart, no line out of order, nothing
foreign written into it - which is the shape a single trainer from 12:55:56 leaves. That is weaker
than exclusivity, and it is what the evidence supports. The discarded arms left no artifact either
way: their checkpoint interval was 500k and they ran three minutes.

**The fix is a lock held across the interval, and its first test is 129 minutes long.** A chain now
announces its pid in one shared `%TEMP%\machine.lock` and holds it until it is done; the process scan
survives only to clear a crashed predecessor. `chain23.log` shows the result: `machine held by pid
50180, waiting` once a minute from 12:55:48 to 15:03:50, then `machine claimed` at 15:04:50 - 37 s
after chain22's last evaluation finished. The same scan that had passed 2 h 9 min earlier would have
let that chain start a second trainer next to a running one; the lock is what made it wait. One more
hole stays open: a lock only binds the processes that take it, and the copy already running when the
patch was written had parsed the unpatched script, so relaunching a chain is not the same as retiring
the one before it.

## The reading a single draw invites, and the design that refused it

`euler` went 5M at seed 7 and produced a flat line: mean 3,150.93 falling to -76.12 over five
checkpoints, forward speed ~0, 0.10 falls per episode, target reached in 0 of 100 scored episodes. The
sentence that writes itself from that is "`euler` does not learn this task", and it would have been
published at 04:47 with a gate pinning every cell of it - the gate would have passed, because the
numbers are real, and the claim would still have been wrong. Seed 8 in the same world at the same
budget climbed to 14,409.67 at 4M and reached in 10% of episodes at 5M.

**Why the second draw existed before the first one finished:** the same evening the published world had
swung from +18,956.17 with 10% reaching to -763.96 with 0% on a seed change alone, so a single arm of
this trainer could no longer be quoted as a property of anything but itself. The paired design was
bought for the physics comparison and it paid out on the euler arm instead: the cheap-in-drift world now
has a two-draw record ({0%, 10%}) that is the same multiset as the published world's ({10%, 0%}), which
is a sentence no single draw can support in either direction.

**How to apply:** when an arm is one run of a stochastic trainer, the paragraph has to say which of the
two claims it is making - the ordering (replicable across draws, cheap to test) or the level (not). If
the paragraph needs the level, the second draw is part of the experiment, not polish, and the right time
to schedule it is before the first one finishes, while the interpretation is still open.

## The throughput knob was the optimisation-dose knob

`--num-envs` is named for parallelism and reads like a pure throughput setting. In this SAC loop it is
also the learning dose: there is one critic update per collection iteration
(`train_walker.py:1902-1964`) and a collection iteration is `num_envs` environment steps, so the
optimisation applied to each environment step is 1/`num_envs`. The screens in the physics-preset
section run at 8 envs and the flagship 40M run at 32 - same step budget, four times the gradient
steps. The rates say the same thing from the other side: 1,798.4 s per 1M at 8 envs against 467.4 s
per 1M at 32 is 3.85x, which is the 4x update ratio arriving in the wall clock, not an engineering
gain. Asking "why is 5M cheaper at 32 envs" has the answer "because it trains less per step", and any
comparison of a small-`n` arm against a large-`n` one at equal environment steps is comparing two
different amounts of training.

The trainer now has the explicit knob (`--utd-ratio`, G updates per collection iteration, default 1 so
every committed run is untouched), and two of the blocks that moved matter more than the loop itself.
The Polyak target sync went **inside** it: a target network's lag is measured in gradient steps, so
applying tau once per collection iteration at ratio 4 would leave the target four times further behind
the critic than ratio 1 does - a second semantics change hidden inside the first, invisible at ratio 1
where both readings are identical. The loss logging stayed **outside**, because its cadence is the
tensorboard series the training-rate summaries read. REDQ already had this knob as `--utd-ratio 20`,
and its loop carries the same comment about gating on `global_step` inside a UTD loop - the pattern
was documented in one trainer and quietly absent from the two that ran most of the research.

## A rate published from one window was refuted by the next window of the same recipe

The PPO re-run at 10M cost **1,594 s** in chain25's window, and the paragraph written from it said the
whole curve costs less clock than a SAC 5M arm in this world. Chain29 ran the identical recipe - same
config, seed 8 - and took **3,636 s**: 2,739.7 env-steps/s where chain25 measured 6,249.4, a **2.28x**
spread, both windows on AC power, and 3,636 s is more than either SAC ratio-1 arm (2,399 s, 2,529 s)
and less than either ratio-4 arm (6,494 s, 6,824 s). The published sentence was true of a measurement
and false as a property, and the only thing that caught it was running the same thing again.

**Why this one stings more than the usual per-window noise:** SAC's four 5M arms across two nights
disagreed by 5.4% in seconds per 1M at fixed dose (479.8 and 505.8), and its dose ratio replicated to
0.3% (2.707 and 2.698). PPO's own rate moved 2.28x. A trainer whose update is one big vectorised pass
per rollout appears to read the machine's state far more strongly than one whose update is a small
batched step - so "PPO is Nx faster than SAC" is a claim about two windows as much as about two
algorithms, and here neither PPO window shared a machine moment with a SAC arm at all, so an honest
within-window PPO-vs-SAC ratio does not exist in this repository yet.

**How to apply:** a rate gets the window it was measured in named next to it, and a cross-algorithm rate
needs both arms in one window or it is not a ratio, it is a coincidence of two. Where the honest ratio
cannot be built, publish the straddle ("comparable, between SAC's ratio-1 and ratio-4 arms") and record
what would settle it - an idle-gap replication of chain25, to see whether the fast window is reproducible
at all.

## An episode mean cannot tell a walk from a fall forward

The README said the agent "walks", and the evidence was two numbers: `mean_x_velocity` per episode, and
`reached_target_pct` per checkpoint. The user watched the rendered clips and said the thing they saw was
an agent trying to stand up. Both readings cannot be about the same trajectory, so the trace was split per
step at whether the torso was inside the standing band (`bench_approach_mechanism.py`, which replays the
same 280 published episodes and refuses to write unless each reproduces its committed telemetry). In the
published world, the 35 episodes the distance rule counts as reaches close 46.61 m at in-band steps and
43.43 m below the band; 30 of their 35 closest approaches come after the episode's first fall; and the
longest continuous in-band run in any reaching episode is 1.47 s, with none reaching two seconds. The
verb was doing work the metric could not carry.

**Why the mean was so misleading:** it integrates over exactly the distinction being asked about. An
episode that walks 3 m at 0.4 m/s and an episode that topples forward twice and covers the same ground
both report `mean_x_velocity = 0.4`. Nothing in the episode-level record separates them, so the sentence
"the arriving group averages 0.36-0.42 m/s" was read as locomotion while being equally a description of
falling. The stricter arrival rule - torso above 1.0 m and upright above 0.7 *at* the step inside the
radius - fixes what counts as arriving and still says nothing about how the body travelled, which is why
13 of the 16 upright arrivals in the published world also have their closest approach after a fall.

**How to apply:** when a claim uses a verb that describes a process ("walks", "reaches", "stands",
"recovers"), the metric has to be computed at the time resolution the verb implies - per step for a
process, not per episode for an outcome. If the only available metric is an episode aggregate, say what
it can and cannot express instead of choosing the reading that matches the video. And note which
instrument settled it: not a new experiment, the same fidelity-gated replay one level down.

## The reward knob that was dead in the phase being trained

The experiment designed to turn "advances in bursts" into walking had two arms, and the first one was
`stillness_penalty_weight` 5 to 50 - a lever on the freeze-while-standing failure. It cost nothing to
check before spending 7.6 GPU-hours on it, and the check killed it: `stillness_penalty` is assigned inside
`if task_phase == "balance":` (`envs/walker_ragdoll_env.py:493-498`) and initialised to `0.0` otherwise, so
in the target phase its contribution to every gradient is identically zero. The arm would have been a
duplicate of the control at a different price, and its result would have been read as "raising the
stillness penalty does not help".

The live anti-collapse term in the target phase is `stability_reward_weight` - `standing_gate ×
exp(-0.25 · angular speed) × one foot down with no bad contacts` - and tripling it through the new
`--reward-override` is the first intervention in this repository that moves the strict arrival column:
4 of 20 episodes arrive standing where 3 of 20 had been the ceiling, with the direction replicated on a
second seed. The other arm, the target distance curriculum, is reported as the negative it is: 1 of 20
upright in its own task and 1 of 20 on the published 2-5 m.

**Why this is a class of error and not one mistake:** `TRAINING_REWARD_KWARGS` lists seven weights with
identical names, types and plumbing, and one of them does nothing in the phase being trained. Nothing in
the dict distinguishes a live knob from a dead one, and the evaluator scores a run trained under either.
The same asymmetry hid in the environment's own vocabulary: `_curriculum_level` counted targets reached
for the whole life of this repository and nothing read it, which is why the "curriculum" in
`standup_balance_walk_curriculum_v9` was resampling alone until this week.

**How to apply:** before pricing an arm, read the branch the term lives in and the phase guard around it.
The README's "this arm was dead" sentence is now gated on the source - the cited range has to contain the
assignment *and* the `balance` guard, and no second non-zero assignment may exist outside it - so the day
the term goes live in target, the claim fails loudly instead of rotting quietly. Related:
[[judge-a-result-against-the-task-s-own-success-criterion]],
[[an-episode-mean-cannot-carry-a-verb]].

## The planner arrives and does not walk: what a simulator in the loop is actually worth

The residual-learning proposal assumed the missing piece was a controller good enough to correct. Measured
against one, that assumption splits. `bench_mpc.py` runs a receding-horizon cross-entropy planner on the
same 20 seeded target episodes, given the true dynamics and the same shaped reward the learned policies
optimise, and scores it with the same per-step trace. At a small search budget it reaches 3 of 20 and
arrives standing in 3 of 20 - the published ceiling. With more search (48 samples × 3 iterations, 1.0 s
horizon) it reaches **9 of 20 and arrives standing in 6 of 20**, above the best learned arm's 4 of 20. On
the gait measure it is the worst thing in the comparison: it closes **27.9%** of its ground inside the
standing band, where the published learned pool closes 51.8% and the stability arm 71.3%, and its longest
continuous stand is 1.40 s against their 1.47 s.

**Why this is a split and not a ranking:** arrival is a property of the endpoint, and search is very good
at endpoints - it can spend its whole budget finding the one state inside the radius with the torso up.
Walking is a property of the path, and the planner's path is a series of recoveries, because the reward it
is given zeroes every locomotion term below the standing band and pays posture the moment the torso is up.
The same objective that makes the learned policies fall forward makes the search prefer falling forward.

**The reproducibility surprise, which is the part worth keeping.** Two independent draws of the small cell
agree exactly on reach, upright arrivals, band time and share closed standing - even though the open loop
is not deterministic: two rollouts from a byte-identical restored state diverge within a few steps, because
the constraint solver is not bit-reproducible. Closed-loop behaviour reproduced while open-loop trajectories
did not, so "the solver is non-deterministic" is not by itself a reason to distrust a planner row - the
draw is what settles it, and here it settled.

**How to apply:** before treating a planner as the base a learned residual will fix, measure the planner on
the same instrument as the learner and look at the path column, not the arrival column. If the planner wins
arrival and loses gait, a residual on top of it inherits the gait problem, and the cheaper question is why
the reward makes falling the cheapest way to move.

## Before building the student, ask whether the teacher is predictable

The planner-as-teacher line was built end to end - collector, BC trainer on the RL arms' own architecture,
two cells (raw and in-band-filtered), scoring on the published protocol with disjoint seeds - and both
students reached in 0 of 20. The diagnosis that explains it takes about a minute of numpy and would have
been worth running first: on an episode-level split of the demonstrations, predict the teacher's action.
A constant zero gives validation MSE 0.3753, the per-dimension mean 0.3751, ridge 0.3758, nearest-neighbour
0.750 at k=1 down to 0.387 at k=32, and the trained students 0.373 and 0.392. Nothing beats the constant.
The reason is in the signal itself: consecutive recorded actions differ by MSE 0.7567 while the action's own
energy is 0.3806, because the executed action is the argmin of a stochastic search and flips between
near-tied sequences. Replacing it with the elite-set mean - the standard MPC policy-extraction move - moves
the jitter to 0.7408, still loses to the constant on the same test, and costs the teacher its arrival rate
(1 of 4 reaching, down from 12 of 28).

**Why the order matters:** every stage of the pipeline was individually sound - the seeds were disjoint, the
split was by episode, the checkpoint loaded through the same scorer, the best-epoch weights were kept after
the first pass showed validation worsening from epoch 1. None of that can recover a target that does not
depend on the input. The predictability test is a property of the *teacher's output distribution* and is
measured on data already collected, so it costs nothing once the demonstrations exist and everything once
they do not.

**How to apply:** when a teacher is proposed - a planner, an expert policy, a dataset from elsewhere - run
the constant-vs-ridge-vs-k-NN probe on its actions before training a student. If no regressor separates
itself from the mean, the project is not "train a better student", it is "the teacher's decisions are not a
function of what the student can see", and the fix belongs on the teacher's extraction or the observation,
not on the loss. Related: [[an-episode-mean-cannot-carry-a-verb]],
[[quantify-cost-dont-claim-impossibility]].
