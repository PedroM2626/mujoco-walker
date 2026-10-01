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
    "note": "Phase 1 previously had no recorded results at all. These runs are short by "
            "design (minutes, not the 1M-step budget); they show the trainers execute and "
            "receive reward, and are not comparable with the Phase 3/4 tables.",
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

print(json.dumps(data, indent=2, ensure_ascii=False))
