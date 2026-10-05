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
