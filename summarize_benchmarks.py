"""Build machine-readable summaries of the re-measured Phase 3 and Phase 4 benchmarks.

Reads the run logs, so the published numbers cannot drift from what was executed.
Re-run after a benchmark to refresh benchmarks/*.json.
"""
import hashlib
import json
import os
import re
from datetime import datetime as dt

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


def _stats(values):
    arr = np.asarray(values, dtype=float)
    return {
        "episodes": int(arr.size),
        "mean": round(float(arr.mean()), 2),
        "median": round(float(np.median(arr)), 2),
        "std": round(float(arr.std()), 2),
        "min": round(float(arr.min()), 2),
        "max": round(float(arr.max()), 2),
        "q25": round(float(np.percentile(arr, 25)), 2),
        "q75": round(float(np.percentile(arr, 75)), 2),
    }


def phase3(path="eval_phase3_20ep.log", seed=7, flags="", desc=None,
           out="benchmarks/phase3_merging_20ep.json"):
    text = open(os.path.join(ROOT, path), encoding="utf-8", errors="replace").read()
    blocks = re.split(r"--- Evaluating (.+?) ---", text)[1:]
    out_data = {"protocol": None, "models": {}}
    counts = set()
    for i in range(0, len(blocks) - 1, 2):
        name, body = blocks[i], blocks[i + 1]
        rewards = [float(v) for v in re.findall(r"Reward:\s*(-?[\d.]+)", body)]
        falls = [float(v) for v in re.findall(r"Quedas:\s*([\d.]+)", body)]
        # The log is written with the OS code page, so the accented word in
        # "Terminou de pé" arrives mangled; match up to the colon instead of the word.
        standing = re.findall(r"Terminou\s+de\s+\S+\s*:\s*(True|False)", body)
        if rewards and standing and len(standing) != len(rewards):
            raise SystemExit(
                f"{name}: {len(standing)} survival flags for {len(rewards)} episodes - "
                "the log parser and the evaluator have diverged, fix one of them"
            )
        if not rewards:
            continue
        counts.add(len(rewards))
        entry = _stats(rewards)
        # Distribution descriptors, not just the mean: this task is bimodal, and "the merge
        # scored -17k" and "the merge scored -17k because most episodes are falls" are different
        # claims. Recomputed per protocol so a change of scoring can be compared episode-by-
        # episode rather than by re-deriving an outlier argument by hand.
        arr = np.asarray(rewards, dtype=float)
        ordered = np.sort(arr)
        entry["mean_without_best"] = round(float(ordered[:-1].mean()), 2)
        entry["mean_without_best_5"] = round(float(ordered[:-5].mean()), 2)
        entry["std_over_abs_mean"] = round(float(arr.std() / abs(arr.mean())), 2)
        entry["pct_above_20k"] = round(100.0 * float((arr > 20000.0).mean()), 1)
        entry["pct_below_minus_10k"] = round(100.0 * float((arr < -10000.0).mean()), 1)
        entry["falls_per_episode"] = round(float(np.mean(falls)), 2) if falls else None
        entry["standing_at_end_pct"] = round(
            100.0 * sum(1 for s in standing if s.lower() == "true") / max(1, len(standing)), 1
        )
        out_data["models"][name] = entry
    if len(counts) > 1:
        raise SystemExit(f"{path}: strategies ran different episode counts {sorted(counts)}")
    # Derived from the log rather than restated: a hand-written protocol string is how the
    # 100-episode file ended up labelled `--num-episodes 20`.
    out_data["protocol"] = (
        f"evaluate_merging.py --num-episodes {counts.pop() if counts else 0} --seed {seed}{flags} "
        + (desc or "(corrected harness: seeded resets, checkpoint obs_rms applied, survival "
                   "read from env health)")
    )
    return out_data, out


def phase4(path=os.path.join("openai_walker", "eval_phase4_20ep.log"), seed=None,
           out="benchmarks/phase4_race_20ep.json"):
    text = open(os.path.join(ROOT, path), encoding="utf-8", errors="replace").read()
    rows = re.findall(
        r"^(.+?): (-?[\d.]+) Avg Reward \| std ([\d.]+) \| min (-?[\d.]+) \| max (-?[\d.]+)$",
        text, re.M,
    )
    # The log's own header carries the protocol; restating it by hand is how the 50-episode
    # file got labelled `--episodes 20`.
    header = re.search(r"FINAL RESULTS \(headless, (\d+) episodes, seed (\d+)\)", text)
    if not header:
        raise SystemExit(f"{path}: no 'FINAL RESULTS (headless, N episodes, seed S)' header")
    episodes, logged_seed = int(header.group(1)), int(header.group(2))
    if seed is not None and seed != logged_seed:
        raise SystemExit(f"{path}: header says seed {logged_seed}, caller said {seed}")
    out_data = {"protocol": f"evaluate_all.py --episodes {episodes} --seed {logged_seed} "
                            "(headless, seeded per episode)", "models": {}}
    for name, mean, std, mn, mx in rows:
        out_data["models"][name.strip()] = {
            "mean": float(mean), "std": float(std), "min": float(mn), "max": float(mx),
            "episodes": episodes,
        }
    return out_data, out


def phase4_with_extratrees(data):
    """Fold the Extra Trees row into the Phase-4 table.

    It is trained by `train_extratrees.py` and scored by `eval_extratrees.py` rather than by
    `evaluate_all.py`, so it never appeared in that log - which is how a competitive model
    ended up in a footnote. The protocol matches: same environment, same seed+i per episode.
    """
    path = os.path.join(ROOT, "openai_walker", "eval_extratrees_50ep.log")
    if not os.path.exists(path):
        return data
    text = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(r"Extra Trees: (-?[\d.]+) avg over (\d+) ep \(std ([\d.]+), "
                  r"min (-?[\d.]+), max (-?[\d.]+)\)", text)
    if not m:
        return data
    mean, episodes, std, mn, mx = m.groups()
    if int(episodes) != next(iter(data["models"].values()))["episodes"]:
        raise SystemExit(f"Extra Trees ran {episodes} episodes, the race ran "
                         f"{next(iter(data['models'].values()))['episodes']}: not comparable")
    data["models"]["Extra Trees Cloner (sklearn)"] = {
        "mean": float(mean), "std": float(std), "min": float(mn), "max": float(mx),
        "episodes": int(episodes),
    }
    data["protocol"] += " + eval_extratrees_50ep.log (same seeded protocol)"
    return data


N1_LABELS = {
    # `final_results.txt` is captured Portuguese console output from play_race.py; these are the
    # keys the 50-episode per-episode file uses. An explicit map, so a rename in either file fails
    # the builder instead of silently dropping a row from the comparison.
    "Teacher (Upper Bound)": "Teacher (Upper Bound)",
    "Behavioral Cloning Puro": "BC",
    "Implicit Q-Learning": "IQL",
    "Conservative Q-Learning": "CQL",
    "BC+SAC (Naive Initialization)": "BC+SAC (Naive)",
    "BC+SAC (Regularization Penalty)": "BC+SAC (Regularized)",
    "BC+SAC (Action Constraints)": "BC+SAC (Constrained)",
    "CQL+SAC (Offline-to-Online)": "CQL+SAC",
    "Inverse RL (GAIL)": "GAIL",
    "Inverse RL (AIRL)": "AIRL",
    "Batch-Constrained Q-learning (BCQ)": "BCQ",
    "Decision Transformer (DT)": "DT",
    "MaxEnt IRL": "MaxEnt",
}


def phase4_n1_reassessment(out="benchmarks/phase4_n1_vs_50ep.json"):
    """Score every single-episode record against the distribution the same weights produce now.

    `final_results.txt` is one unseeded episode per model, and the README retired it for being a
    poor estimator. That is only half a statement: for some models the old draw sits comfortably
    inside today's spread (it was a real, high episode), for others it sits outside anything the
    recorded protocol can produce (it cannot be episode luck). The distinction decides whether a
    retired number is a measurement of a different thing or a bad estimate of this one, and it is
    arithmetic over two committed files, so it belongs in an artifact rather than in prose.
    """
    text = open(os.path.join(ROOT, "openai_walker", "final_results.txt"),
                encoding="utf-8", errors="replace").read()
    draws = {m: float(v) for m, v in re.findall(
        r"\[(.+?)\] Preparando para a corrida.*?\[\1\] Episodio 1 - Pontuacao: (-?[\d.]+)",
        text, re.S)}
    if not draws:
        raise SystemExit("final_results.txt: no 'Episodio 1 - Pontuacao' rows parsed")
    path = os.path.join(ROOT, "openai_walker", "final_episodes_50ep_seed2026.json")
    episodes = json.load(open(path, encoding="utf-8"))
    models = episodes["models"]
    unmatched = [label for label in draws if label not in N1_LABELS]
    if unmatched:
        raise SystemExit(f"final_results.txt labels with no mapping: {unmatched}")

    out_data = {
        "protocol": "single-episode draws from openai_walker/final_results.txt scored against "
                    f"evaluate_all.py --episodes {episodes['episodes']} --seed {episodes['seed']} "
                    "(per-episode returns in final_episodes_"
                    f"{episodes['episodes']}ep_seed{episodes['seed']}.json)",
        "std_convention": "population std, the same one evaluate_all.py prints and the README "
                          "table quotes",
        "models": {},
    }
    for label, n1 in draws.items():
        name = N1_LABELS[label]
        arr = np.asarray(models[name], dtype=float)
        stats = _stats(models[name])
        z = (n1 - stats["mean"]) / stats["std"] if stats["std"] else None
        entry = dict(stats)
        entry.update({
            "n1_score": n1,
            "n1_in_sigmas": round(float(z), 2) if z is not None else None,
            "episodes_at_or_above_n1": int((arr >= n1).sum()),
            # Episode luck cannot reach outside the band the 50 seeded resets span; a draw that
            # does came from something other than the reset.
            "explainable_as_episode_luck": bool(stats["min"] <= n1 <= stats["max"]),
        })
        out_data["models"][name] = entry
    return out_data, out


GAIL_RETIRED_FIGURE = 1016.41  # the score the README carried before the seeded table's 998.07
GAIL_WEIGHTS = {
    # The two policies this comparison scores. Their bytes are gitignored (everything `*.pt` is), so
    # the hashes and mtimes below are recorded captures; the builder verifies them whenever the
    # files are present on the machine it runs on, and fails rather than quietly re-label them.
    "june_control": {"file": "openai_walker/gail_model.pt",
                     "sha256": "01380EDB3A7759B4F314E029785E4A9072436509AD7E0744F5566D6ECC049DA6",
                     "written": "2026-06-10 19:03"},
    "fresh_retrain": {"file": "openai_walker/gail_model_retrain_2026-10-05.pt",
                      "sha256": "C6CF60D517F11ED539E4A3668CC1029A34DA4CD765BF598CE27706A5C1A7E85E",
                      "written": "2026-10-05 11:26"},
}


def _train_series(path, pattern):
    text = open(os.path.join(ROOT, path), encoding="utf-8", errors="replace").read()
    return [(int(s), float(v)) for s, v in re.findall(pattern, text, re.M)]


def _train_stats(rows, from_step=10000):
    """Episode-return statistics over the window both records can see.

    `start_steps` is 10000 in the trainer: those first steps are uniform-random and mlflow never
    logs their episode returns (`if t >= start_steps`), while a stdout capture prints every episode.
    Taking both series from step 10000 onwards is what makes "the June run" and "the retrain" the
    same measurement rather than two different denominators.
    """
    rows = [r for r in rows if r[0] > from_step]
    values = [v for _, v in rows]
    return {
        "window_from_step": from_step,
        "episodes_logged": len(values),
        "last_step": rows[-1][0],
        "max": round(max(values), 2),
        "mean_of_last_20": round(sum(values[-20:]) / min(20, len(values)), 2),
    }


def gail_retrain(out="benchmarks/phase4_gail_retrain.json"):
    """The 2026-10-05 GAIL retrain, scored against the June policy it was set beside.

    The README's GAIL anomaly rested on arithmetic over one checkpoint: 1016.41 sits above anything
    the 50 seeded resets of `gail_model.pt` produce. A single policy cannot say whether that band is
    a property of GAIL or of that particular run, so a second 1M-step run was trained with the same
    recipe (unseeded, so this is a fresh draw and not a reproduction) and scored under the published
    protocol, with the June weights re-scored in the same session as the control. The training-side
    series is here too, because that is where the retired figure actually lands inside a range:
    `true_env_reward` is the environment return the trainer prints per episode, and it is a
    different quantity from the seeded evaluation mean.
    """
    def eval_arm(name):
        data = json.load(open(os.path.join(ROOT, "openai_walker", name), encoding="utf-8"))
        values = data["models"]["GAIL"]
        stats = _stats(values)
        stats["band"] = round(stats["max"] - stats["min"], 2)
        return stats

    def weights(label):
        spec = GAIL_WEIGHTS[label]
        entry = dict(spec, present=False)
        path = os.path.join(ROOT, *spec["file"].split("/"))
        if os.path.exists(path):
            entry["present"] = True
            entry["sha256_on_disk"] = hashlib.sha256(open(path, "rb").read()).hexdigest().upper()
            if entry["sha256_on_disk"] != spec["sha256"]:
                raise SystemExit(f"{spec['file']}: hash {entry['sha256_on_disk']} is not the "
                                 f"{spec['sha256']} this artifact was written against")
            entry["mtime_on_disk"] = dt.fromtimestamp(
                os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
        return entry

    fresh = eval_arm("final_episodes_50ep_seed2026_gail_retrain.json")
    control = eval_arm("final_episodes_50ep_seed2026_gail_june_control.json")
    published = json.load(open(os.path.join(ROOT, "benchmarks", "phase4_race_50ep.json"),
                               encoding="utf-8"))["models"]["GAIL"]

    fresh_rows = _train_series(
        os.path.join("openai_walker", "gail_retrain_2026-10-05.log"),
        r"^Step (\d+) \| Episode \d+ \| True Env Reward: (-?[\d.]+)$")
    fresh_train = _train_stats(fresh_rows)
    june_record = json.load(open(os.path.join(ROOT, "benchmarks",
                                              "gail_june_run_true_env_reward.json"),
                                 encoding="utf-8"))
    june_rows = [(e["step"], e["true_env_reward"]) for e in june_record["episodes"]]
    june_train = _train_stats(june_rows)

    above = [e for e in june_record["episodes"] if e["true_env_reward"] >= GAIL_RETIRED_FIGURE]
    nearest = min(june_record["episodes"],
                  key=lambda e: abs(e["true_env_reward"] - GAIL_RETIRED_FIGURE))
    fresh_above = [r for r in fresh_rows if r[1] >= GAIL_RETIRED_FIGURE]

    return {
        "protocol": "evaluate_all.py --episodes 50 --seed 2026 (headless, seeded per episode), "
                    "run once per policy in the same session",
        "std_convention": "population std, the same one evaluate_all.py prints and the README "
                          "table quotes",
        "retired_figure": GAIL_RETIRED_FIGURE,
        "arms": {
            "june_control": {"evaluation": control, "training_episodes": june_train,
                             "training_run": {"run_id": june_record["run_id"],
                                              "wall_clock_seconds": june_record["wall_clock_seconds"],
                                              "params": june_record["params"]},
                             "weights": weights("june_control")},
            "fresh_retrain": {"evaluation": fresh, "training_episodes": fresh_train,
                              "training_run": {"wall_clock_seconds": 11897,
                                               "wall_clock_source": "mlruns.db run "
                                                   "95f7940d8c944425a398a59291d56837 (local and "
                                                   "gitignored): 2026-10-05 08:08:35 -> 11:26:52",
                                               "params": {"algorithm": "GAIL",
                                                          "max_steps": "1000000",
                                                          "batch_size": "256"}},
                              "weights": weights("fresh_retrain")},
        },
        # The control is the same weights, the same seed and the same code path as the published
        # table, so any difference here is a change in the harness, not in the policy.
        "control_reproduces_published": {
            "published": {k: published[k] for k in ("mean", "std", "min", "max")},
            "measured": {k: control[k] for k in ("mean", "std", "min", "max")},
            "exact": all(control[k] == published[k] for k in ("mean", "std", "min", "max")),
        },
        "comparison": {
            "mean_delta_june_minus_fresh": round(control["mean"] - fresh["mean"], 2),
            "band_ratio_fresh_over_june": round(fresh["band"] / control["band"], 2),
            "retired_above_fresh_max": round(GAIL_RETIRED_FIGURE - fresh["max"], 2),
            "retired_above_june_max": round(GAIL_RETIRED_FIGURE - control["max"], 2),
            # The retired figure is outside both evaluations and inside the June run's training
            # episodes, which is what a number read off the wrong column would look like.
            "june_training_episodes_at_or_above_retired": len(above),
            "fresh_training_episodes_at_or_above_retired": len(fresh_above),
            "nearest_june_training_episode": {
                "step": nearest["step"], "value": nearest["true_env_reward"],
                "delta": round(abs(nearest["true_env_reward"] - GAIL_RETIRED_FIGURE), 2)},
        },
    }, out


TEACHER_EVENTS = os.path.join("openai_walker", "sac_walker_tensorboard", "SAC_2",
                              "events.out.tfevents.1780843517.pedro.36828.0")


def teacher_eval_curve(out="benchmarks/teacher_eval_curve.json"):
    """The teacher's own training-time evaluations, mined out of its committed TensorBoard log.

    Worth extracting for two reasons. It is the only evaluation series in the repository from
    before the retracted README, and its protocol is *independent* of `evaluate_all.py`: SB3's
    `EvalCallback` defaults to 100 deterministic episodes per point, unseeded, on a freshly made
    `Walker2d-v5`. So it is the closest thing to a second opinion on the teacher's level - and
    where the two protocols disagree is information, not noise to average away.

    It also settles a negative: these are the teacher's numbers, not the offline policies', so
    nothing in the repository ever logged a score for BC, BCQ or GAIL before 2026-06-26.
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator as ea
    except ImportError:
        raise SystemExit("teacher_eval_curve needs tensorboard to read the event file")
    path = os.path.join(ROOT, TEACHER_EVENTS)
    if not os.path.exists(path):
        raise SystemExit(f"{TEACHER_EVENTS} is missing; the curve cannot be rebuilt")
    acc = ea.EventAccumulator(path)
    acc.Reload()
    if "eval/mean_reward" not in acc.Tags().get("scalars", []):
        raise SystemExit(f"{path} has no eval/mean_reward scalars")
    rows = [{"step": s.step, "mean_reward": round(float(s.value), 2),
             "mean_ep_length": round(float(s.value), 2)} for s in acc.Scalars("eval/mean_reward")]
    lengths = [round(float(s.value), 2) for s in acc.Scalars("eval/mean_ep_length")]
    for row, length in zip(rows, lengths):
        row["mean_ep_length"] = length
    late = [r["mean_reward"] for r in rows if r["step"] >= 400_000]
    return {
        "source": TEACHER_EVENTS.replace(os.sep, "/"),
        "protocol": "SB3 EvalCallback during train_teacher.py: eval_freq=10000, "
                    "n_eval_episodes default 100, deterministic=True, unseeded resets",
        "evaluations": rows,
        "after_400k_steps": {"n": len(late), "min": min(late), "max": max(late),
                              "mean": round(sum(late) / len(late), 2)},
        "final_eval": rows[-1]["mean_reward"],
        "note": "The scored teacher (`sac_walker2d_final.zip`) measures 3516.95 mean over 50 "
                "seeded episodes in the Phase-4 table; its final 100-episode training-time "
                "evaluation is above. The two protocols differ in episodes-per-point, seeding "
                "and which checkpoint is in play (EvalCallback also wrote logs/best_model.zip), "
                "so the gap is an open question, not a number to reconcile by averaging.",
    }, out


BENCHMARKS = [
    phase3,
    lambda: phase3("eval_phase3_100ep.log", seed=11,
                   out="benchmarks/phase3_merging_100ep.json"),
    # Same harness re-run after the reward-protocol fix: scored with the reward the experts were
    # trained against, in env v9. Kept beside the retired file so the two protocols can be
    # compared episode-by-episode instead of argued about.
    lambda: phase3("eval_phase3_100ep_v9.log", seed=11,
                   desc="(env v9, scored with the training shaping: the reward the experts "
                        "actually optimise. See benchmarks/reward_term_breakdown_v9.json for "
                        "what the two protocols differ by per step)",
                   out="benchmarks/phase3_merging_100ep_v9_trainreward.json"),
    lambda: phase3("eval_phase3_100ep_rawobs.log", seed=11, flags=" --raw-obs",
                   desc="(obs_rms deliberately ignored: this arm reproduces the pre-fix "
                        "harness, so it is a measurement of the bug, not a score)",
                   out="benchmarks/phase3_merging_100ep_rawobs.json"),
    phase4,
    lambda: (phase4_with_extratrees(
        phase4(os.path.join("openai_walker", "eval_phase4_50ep.log"), seed=2026,
               out="benchmarks/phase4_race_50ep.json")[0]),
        "benchmarks/phase4_race_50ep.json"),
    # Where each retired single-episode draw sits inside the distribution the same weights
    # produce today, so "the old table was a poor estimator" can be said per model.
    phase4_n1_reassessment,
    # A second 1M-step GAIL run against the June one, so the width of the anomaly is measured
    # rather than argued: evaluation spread within a run, and run-to-run spread of the mean.
    gail_retrain,
    # The teacher's own 100-episode evaluations, mined from its committed TensorBoard log: the only
    # pre-retraction evaluation series in the repo, and an independent protocol.
    teacher_eval_curve,
]


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "benchmarks"), exist_ok=True)
    for builder in BENCHMARKS:
        try:
            data, target = builder()
        except FileNotFoundError as e:
            print(f"skipped: {e}")
            continue
        with open(os.path.join(ROOT, target), "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
        models = data.get("models", {})
        print(f"{target}: {len(models)} models" if models else f"{target}")
        for name, s in models.items():
            print(f"   {name:32s} mean={s['mean']:>10} median={s.get('median', '-'):>10} "
                  f"std={s['std']}")
