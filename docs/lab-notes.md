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
