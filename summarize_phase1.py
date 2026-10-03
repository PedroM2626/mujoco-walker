"""Summarise the short Phase-1 (DreamerV3 / REDQ / ARS) evidence runs.

These are deliberately labelled smoke-length: they establish that the trainers run and
learn on the ragdoll, not benchmark numbers. Regenerates benchmarks/phase1_evidence.json
from the logs so nothing is transcribed by hand.
"""
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


def returns_from(path):
    text = open(os.path.join(ROOT, path), encoding="utf-8", errors="replace").read()
    return [float(v) for v in re.findall(r"episodic_return=(-?[\d.]+)", text)], text


NOTE = (
    "Phase 1 previously had no recorded results at all. These runs are short by "
    "design (minutes, not the 1M-step budget); they show the trainers execute and "
    "receive reward, and are not comparable with the Phase 3/4 tables. SCALE "
    "WARNING: REDQ and Dreamer wrap the environment in NormalizeReward(gamma), so "
    "the 'episode_return' fields below are normalised sums, not raw environment "
    "returns. On the raw scale the same checkpoints are indistinguishable from an "
    "inert robot: over one seeded 300-step rollout in env v8, commanding zero "
    "scores -3775.76, uniform random -4146.46, REDQ -4248.46, Dreamer -3588.23 "
    "and ARS -4105.24, and none of the five moves more than 0.4 m "
    "(benchmarks/phase1_inert_reference_v8.json). The same rollouts in env v9, "
    "where posture is a bonus rather than a punishment, land in one band from "
    "+237.06 to +1691.21 with the same sub-0.4 m displacements and standing_gate "
    "0.000 throughout (benchmarks/phase1_inert_reference_v9.json) - which is why "
    "the behaviour claim is pinned on displacement, not on the return "
    "(tests/test_phase1_behaviour.py). Reproduce with `python "
    "bench_inert_reference.py --tag v9` and `--tag v8 --env-commit 2d59b7b`; the "
    "Dreamer row also needs the policy RNG seeded, which is why eval_phase1.py "
    "now does it per model. Superseded: an earlier revision of this note quoted "
    "-3826.77 / -3836.79 / -3887.55 / -3857.06, figures transcribed by hand that "
    "could not be reproduced from any recorded protocol."
)


def summary(log, budget_key="Training completed. Total steps:"):
    values, text = returns_from(log)
    completed = re.search(re.escape(budget_key) + r"\s*(\d+)", text)
    out = {"log": log, "episodes_recorded": len(values)}
    if completed:
        out["steps_completed"] = int(completed.group(1))
    if values:
        arr = np.asarray(values)
        out["episode_return"] = {
            "mean": round(float(arr.mean()), 2),
            "median": round(float(np.median(arr)), 2),
            "max": round(float(arr.max()), 2),
            "last10_mean": round(float(arr[-10:].mean()), 2),
        }
    return out


data = {
    "note": NOTE,
    "date": "2026-10-01",
    "runs": {
        "train_ars": summary("ars_evidence.log"),
        "train_redq": summary("redq_evidence.log"),
        "train_dreamer": summary("dreamer_smoke.log"),
    },
}

# ARS reports its own periodic policy evaluations, which are the more meaningful signal
text = open(os.path.join(ROOT, "ars_evidence.log"), encoding="utf-8", errors="replace").read()
evals = [(int(e), float(r)) for e, r in re.findall(r"epoch=(\d+), eval_return=(-?[\d.]+)", text)]
if evals:
    data["runs"]["train_ars"]["eval_returns"] = {
        "epochs_observed": len(evals),
        "first": evals[0][1],
        "last": evals[-1][1],
        "max": max(r for _, r in evals),
        "max_at_epoch": max(evals, key=lambda t: t[1])[0],
    }

with open(os.path.join(ROOT, "benchmarks", "phase1_evidence.json"), "w", encoding="utf-8") as h:
    json.dump(data, h, indent=2, ensure_ascii=False)
    h.write("\n")     # the committed file ends with a newline; regenerating must be a no-op

print(json.dumps(data, indent=2, ensure_ascii=False))
