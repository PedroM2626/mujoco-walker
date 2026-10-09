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


def _curve_rows(data):
    """Every per-episode telemetry row of a curve artifact, as a sorted comparable tuple.

    Sorted because the two files name their arms differently (`s1M` in the published curve,
    `s1000000` in the new one) and the comparison that matters is the set of episodes, not the key.
    """
    return sorted(
        (row["steps"], row["min_target_distance"], bool(row["reached_target"]),
         round(float(row["mean_x_velocity"]), 6))
        for key, rows in data["per_episode"].items() if key.endswith("__telemetry") for row in rows
    )


def target_from_scratch_paired(out="benchmarks/target_learning_curve_from_scratch_paired.json"):
    """Compare the from-scratch target run to the curriculum run episode by episode.

    The two curves share everything that a comparison needs: the same ten checkpoint budgets, the
    same 20 resets (episode *i* is seeded `11+i` in both), the same env version, the same device and
    thread pin, because chain14 re-scored the curriculum arm in the same session rather than trusting
    the published curve, whose device was never recorded. So the difference is a paired sample and not
    two means, and the honest statistic is per-episode: how often did one arm get closer to the
    target than the other, on the same initial state.
    """
    scratch = json.load(open(os.path.join(ROOT, "benchmarks",
                                          "target_learning_curve_from_scratch_v9.json"),
                              encoding="utf-8"))
    curric = json.load(open(os.path.join(ROOT, "benchmarks",
                                         "target_learning_curve_curriculum_rescored_v9.json"),
                             encoding="utf-8"))
    arms = sorted((k for k in scratch["per_episode"] if not k.endswith("__telemetry")),
                  key=lambda k: int(k[1:]))
    deltas, reached = [], {"both": 0, "scratch_only": 0, "curriculum_only": 0, "neither": 0}
    scratch_dist, curric_dist = [], []
    for arm in arms:
        rows_s = scratch["per_episode"][arm + "__telemetry"]
        rows_c = curric["per_episode"][arm + "__telemetry"]
        if len(rows_s) != len(rows_c):
            raise SystemExit(f"{arm}: {len(rows_s)} scratch episodes vs {len(rows_c)} curriculum")
        for a, b in zip(rows_s, rows_c):
            deltas.append(a["min_target_distance"] - b["min_target_distance"])
            scratch_dist.append(a["min_target_distance"])
            curric_dist.append(b["min_target_distance"])
            if a["reached_target"] and b["reached_target"]:
                reached["both"] += 1
            elif a["reached_target"]:
                reached["scratch_only"] += 1
            elif b["reached_target"]:
                reached["curriculum_only"] += 1
            else:
                reached["neither"] += 1

    def dist(values):
        arr = np.asarray(values, dtype=float)
        return {"median": round(float(np.median(arr)), 3), "mean": round(float(arr.mean()), 3),
                "min": round(float(arr.min()), 3)}

    arr = np.asarray(deltas, dtype=float)
    return {
        "protocol": ("paired per-episode comparison of benchmarks/target_learning_curve_"
                     "from_scratch_v9.json against benchmarks/target_learning_curve_"
                     f"curriculum_rescored_v9.json: {len(arms)} matched checkpoint budgets x "
                     f"{len(arr) // len(arms)} seeded resets, negative delta = the from-scratch arm "
                     "came closer"),
        "instrument": {
            "scratch": {"device": scratch.get("device"), "torch_threads": scratch.get("torch_threads"),
                        "env_version": scratch.get("env_version")},
            "curriculum": {"device": curric.get("device"), "torch_threads": curric.get("torch_threads"),
                           "env_version": curric.get("env_version")},
        },
        # Recorded rather than derived: `checkpoints/` and `mlruns.db` are both gitignored, so the
        # only durable statement of what produced the left column is this block plus the checkpoint
        # files themselves. Verified at build time when the directory is present.
        "from_scratch_run": {
            "run_id": "sac_target_from_scratch_40m",
            "config": {"algo": "sac", "seed": 7, "num_envs": 32, "total_timesteps": 40000000,
                       "task_phase": "target", "reset_mode": "mixed",
                       "target_forward_velocity": 1.2, "init_from": None,
                       "checkpoint_interval": 1000000},
            "wall_clock": "2026-10-05 13:46:54 -> 18:58:31 = 5 h 11 min 37 s",
            "wall_clock_seconds": 5 * 3600 + 11 * 60 + 37,
            "sustained_env_steps_per_s": 2145,
            "checkpoints_present": len(arms),
        },
        # The published curriculum curve, re-scored in the same session: identical telemetry is what
        # licenses the paired column. Without this the comparison would be cuda-against-unknown.
        "curriculum_rescore_matches_published": {
            "episodes_compared": len(_curve_rows(curric)),
            "identical": _curve_rows(curric) == _curve_rows(json.load(open(
                os.path.join(ROOT, "benchmarks", "target_learning_curve_v9_trainreward.json"),
                encoding="utf-8"))),
        },
        "pairs": int(arr.size),
        "scratch_closer": int((arr < -0.005).sum()),
        "curriculum_closer": int((arr > 0.005).sum()),
        "within_5mm": int((np.abs(arr) <= 0.005).sum()),
        "median_delta_m": round(float(np.median(arr)), 3),
        "mean_delta_m": round(float(arr.mean()), 3),
        "closest_approach": {"scratch": dist(scratch_dist), "curriculum": dist(curric_dist)},
        "reached_target": dict(reached, total_pairs=int(arr.size)),
        "per_budget": {
            arm: {
                "scratch_checkpoint": scratch["models"][arm]["checkpoint"],
                "curriculum_checkpoint": curric["models"][arm]["checkpoint"],
                "scratch": {k: scratch["models"][arm][k] for k in
                            ("mean", "reached_target_pct", "mean_min_target_distance",
                             "mean_x_velocity", "standing_at_end_pct", "falls_per_episode")},
                "curriculum": {k: curric["models"][arm][k] for k in
                               ("mean", "reached_target_pct", "mean_min_target_distance",
                                "mean_x_velocity", "standing_at_end_pct", "falls_per_episode")},
            }
            for arm in arms
        },
    }, out


PRESET_SCREEN_WALL_CLOCK = {
    # Recorded, not derived: the launcher log is the only place a SAC run's wall clock exists.
    # `train_walker.py` defines the mlflow helpers and never calls them, so mlruns.db holds the
    # Dreamer and GAIL runs and none of these, and `checkpoints/` says what was collected rather than
    # how long it took. The chain logs are committed as evidence; the artifacts below are what the
    # runs scored, and the checkpoints they name are the durable half of the story.
    "v9": {"seconds": 1345, "window": "2026-10-05 19:30:16 -> 19:52:41"},
    "fast": {"seconds": 1172, "window": "2026-10-05 19:52:41 -> 20:12:13"},
}


def physics_preset_screens(out="benchmarks/physics_presets_screen_paired.json"):
    """Side by side: the same SAC screen trained in the published world and in the cheap one.

    Two things are being compared here and only one of them is a score. The wall clock says what the
    2.7x cheaper environment step is worth inside a real training loop, which is the question the
    presets exist to answer; the scores say whether a 1M-step screen can see the walking behaviour
    at all, and it cannot - the from-scratch curve only begins reaching the target between 5M and
    30M, so both arms measure 0 of 20.
    """
    arms = {}
    for preset in ("v9", "fast"):
        data = json.load(open(os.path.join(ROOT, "benchmarks",
                                           f"physics_presets_screen_{preset}.json"),
                               encoding="utf-8"))
        name = list(data["models"])[0]
        model = data["models"][name]
        clock = PRESET_SCREEN_WALL_CLOCK[preset]
        arms[preset] = {
            "checkpoint": model["checkpoint"],
            "scored_in_version": data.get("scored_in_version"),
            "device": data.get("device"), "torch_threads": data.get("torch_threads"),
            "wall_clock_seconds": clock["seconds"], "wall_clock_window": clock["window"],
            "mean": model["mean"], "std": model["std"], "min": model["min"], "max": model["max"],
            "falls_per_episode": model["falls_per_episode"],
            "reached_target_pct": model["reached_target_pct"],
            "mean_min_target_distance": model["mean_min_target_distance"],
            "mean_x_velocity": model["mean_x_velocity"],
        }
    faster = arms["v9"]["wall_clock_seconds"] / arms["fast"]["wall_clock_seconds"]
    return {
        "protocol": ("SAC 1M, seed 7, num_envs=8, task_phase=target, reset_mode=mixed, "
                     "target_forward_velocity=1.2, trained once per preset; scored by "
                     "eval_phase1.py --num-episodes 20 --seed 11 in the world each arm trained in"),
        "arms": arms,
        "end_to_end_speedup_fast_vs_v9": round(faster, 3),
        "note": ("the environment step itself is 2.707x cheaper at n=8 "
                 "(benchmarks/physics_presets.json), and the training run is 1.15x faster - the "
                 "difference is the learner and the vector-env plumbing, which this preset does not "
                 "touch"),
    }, out


PRESET_SCREEN_5M_WALL_CLOCK = {
    # Recorded, not derived: chain18's launcher log, the same two arms back to back in one window.
    "v9": {"seconds": 8992, "window": "2026-10-05 21:05:04 -> 23:34:56"},
    "fast": {"seconds": 7299, "window": "2026-10-05 23:34:56 -> 2026-10-06 01:36:35"},
}


def physics_preset_screens_5m(out="benchmarks/physics_presets_screen5m_paired.json"):
    """The same screen at the smallest budget where the behaviour is visible: 5M, both worlds.

    The 1M pair could not answer the question - neither world reached the target in any of 20
    episodes, and the 40M curve only starts reaching between 5M and 30M. At 5M both worlds do reach,
    which is the result that makes the cheap one usable for this task: the arm trained in `fast`
    reaches in 2 of 20 episodes at 4M and 2 of 20 at 5M while the published world reaches at 5M, so
    the qualitative answer agrees, and the wall clock says what it costs to get it. One draw per arm:
    the two runs differ in world, not in seed, and nothing here separates world from luck.
    """
    arms = {}
    for preset in ("v9", "fast"):
        data = json.load(open(os.path.join(ROOT, "benchmarks",
                                           f"physics_presets_screen5m_{preset}.json"),
                               encoding="utf-8"))
        clock = PRESET_SCREEN_5M_WALL_CLOCK[preset]
        arms[preset] = {
            "device": data.get("device"), "torch_threads": data.get("torch_threads"),
            "scored_in_version": data.get("scored_in_version"),
            "wall_clock_seconds": clock["seconds"], "wall_clock_window": clock["window"],
            "seconds_per_1m": round(clock["seconds"] / 5.0, 1),
            "by_budget": {
                arm: {k: model[k] for k in ("mean", "std", "reached_target_pct",
                                             "mean_min_target_distance", "mean_x_velocity",
                                             "standing_at_end_pct", "falls_per_episode")}
                for arm, model in sorted(data["models"].items(), key=lambda kv: int(kv[0][1:]))
            },
        }
    faster = arms["v9"]["wall_clock_seconds"] / arms["fast"]["wall_clock_seconds"]
    return {
        "protocol": ("SAC 5M, seed 7, num_envs=8, task_phase=target, reset_mode=mixed, "
                     "target_forward_velocity=1.2, one run per world, checkpointed every 1M; each "
                     "arm scored with eval_phase1.py --num-episodes 20 --seed 11 in the world it "
                     "trained in"),
        "arms": arms,
        "end_to_end_speedup_fast_vs_v9": round(faster, 3),
        "window_caveat": ("the 1M pair of this same comparison measured 1.148x in its own window and "
                          "this pair measures 1.232x in this one; the v9 arm alone ran 1,345 s per 1M "
                          "there and 1,798.4 s per 1M here, which is the machine's state and not the "
                          "worlds - only the ratios inside a pair are reusable"),
    }, out


SECOND_DRAW_5M_WALL_CLOCK = {
    # Recorded from chain23's launcher log, the same two worlds back to back in one window at the
    # second seed. The log is committed (`chain23_evidence.log`) and a provenance test re-derives
    # these spans from it, because a SAC run writes no mlflow record to check against.
    "v9": {"seconds": 9386, "window": "2026-10-06 15:05:50 -> 17:42:16", "scoring_seconds": 78},
    "fast": {"seconds": 7568, "window": "2026-10-06 17:43:34 -> 19:49:42", "scoring_seconds": 52},
}


def physics_preset_second_draw(out="benchmarks/physics_presets_screen5m_draws.json"):
    """The 5M preset pair drawn twice, so the ordering can be told from the luck.

    The first draw said the two worlds agree on the question (each reaches the target at 5M) and
    could say nothing about level, because there was one run per world. Two draws per world at two
    seeds separate the two claims, and they come out differently. The ordering replicates: the cheap
    world is ahead at 9 of the 10 world-x-budget cells. The level does not: `v9` scored 18,956.17 at
    one seed and -763.96 at the other, a swing bigger than the gap between the worlds it is being
    compared across, and the reach column went from 10% to 0 of 20. A single 5M draw in the published
    world can therefore produce a policy that never reaches the target at all.
    """
    draws = {}
    for seed, files, clocks in (("seed7", "physics_presets_screen5m_%s.json", PRESET_SCREEN_5M_WALL_CLOCK),
                                ("seed8", "physics_presets_screen5m_seed8_%s.json", SECOND_DRAW_5M_WALL_CLOCK)):
        arms = {}
        for preset in ("v9", "fast"):
            data = json.load(open(os.path.join(ROOT, "benchmarks", files % preset), encoding="utf-8"))
            clock = clocks[preset]
            arms[preset] = {
                "seed": 7 if seed == "seed7" else 8,
                "checkpoint_run": data["models"]["s1000000"]["checkpoint"].split("/")[1],
                "device": data.get("device"), "torch_threads": data.get("torch_threads"),
                "scored_in_version": data.get("scored_in_version"),
                "wall_clock_seconds": clock["seconds"], "wall_clock_window": clock["window"],
                "seconds_per_1m": round(clock["seconds"] / 5.0, 1),
                "scoring_seconds": clock.get("scoring_seconds"),
                "by_budget": {
                    budget: {k: model[k] for k in ("mean", "std", "reached_target_pct",
                                                   "mean_min_target_distance", "mean_x_velocity",
                                                   "standing_at_end_pct", "falls_per_episode")}
                    for budget, model in sorted(data["models"].items(),
                                                key=lambda kv: int(kv[0][1:]))
                },
            }
        faster = (arms["v9"]["wall_clock_seconds"] / arms["fast"]["wall_clock_seconds"])
        draws[seed] = {"arms": arms, "end_to_end_speedup_fast_vs_v9": round(faster, 3)}

    budgets = [f"s{m}000000" for m in range(1, 6)]
    ahead = sum(1 for seed in draws.values() for b in budgets
                if seed["arms"]["fast"]["by_budget"][b]["mean"]
                > seed["arms"]["v9"]["by_budget"][b]["mean"])
    swing = {
        preset: round(max(d["arms"][preset]["by_budget"]["s5000000"]["mean"] for d in draws.values())
                      - min(d["arms"][preset]["by_budget"]["s5000000"]["mean"] for d in draws.values()), 2)
        for preset in ("v9", "fast")}
    return {
        "protocol": ("SAC 5M, num_envs=8, task_phase=target, reset_mode=mixed, "
                     "target_forward_velocity=1.2, one run per world per seed (7 and 8), "
                     "checkpointed every 1M; each arm scored with eval_phase1.py --num-episodes 20 "
                     "--seed 11 in the world it trained in. Same physics preset within a draw, so "
                     "the two worlds differ only in the environment they were trained and scored in"),
        "draws": draws,
        "fast_ahead_cells_of_10": ahead,
        "mean_swing_between_draws_at_5m": swing,
        "reached_target_pct_at_5m": {
            preset: [d["arms"][preset]["by_budget"]["s5000000"]["reached_target_pct"]
                     for d in (draws["seed7"], draws["seed8"])] for preset in ("v9", "fast")},
        "note": ("the ordering is the part that replicated and the level is the part that did not; "
                 "the two wall-clock speedups (1.232x and 1.240x) are the part that was stable all "
                 "along, because they are ratios measured inside one window"),
    }, out


UTD_5M_WALL_CLOCK = {
    # Recorded from chain24's launcher log: the two arms ran back to back in one window, which is
    # what makes their ratio the quotable figure and the comparison against the committed 8-env arm
    # (chain18, a different night) the one that needs the caveat.
    "n32_r1": {"seconds": 2399, "scoring_seconds": 81,
               "window": "2026-10-06 20:15:05 -> 20:55:04"},
    "n32_r4": {"seconds": 6494, "scoring_seconds": 84,
               "window": "2026-10-06 20:56:25 -> 22:44:39"},
}


def utd_dose_pair(out="benchmarks/utd_dose_pair.json"):
    """What it costs to buy the same optimisation dose at 32 environments instead of 8.

    `--utd-ratio` 4 at `num_envs=32` performs the same number of gradient updates per environment
    step as the committed screens at `num_envs=8` ratio 1, so the two arms of this pair differ in
    collection plumbing and update count, not in how hard each environment step is optimised.

    The clock says the update is most of the bill but not all of it: ratio 4 took 2.707x ratio 1 in
    the same window, not 4x. Solving the two measurements as C + U = 2,399 and C + 4U = 6,494 puts
    1,365 s in updates and 1,034 s in collection, i.e. **57%** of a ratio-1 run at 32 envs is the
    gradient step. The Dreamer loop split reaches 78.6% by instrumenting one iteration of a different
    trainer, so these are two methods agreeing on a direction - the update is the larger half - and
    not one number confirming another.

    The score columns disagree with each other in the way this repository has learned to expect: the
    under-dosed arm posted the higher mean return (36,623.52 against 24,137.34) while the dose-matched
    arm reached the target in 20% of episodes at 5M and 25% at 3M against 5% and 15%. The return
    column is posture; the reach column is the task. One draw per arm, and today's own measurement put
    the within-world band at ~20k of mean and 0-10% of reach, so the behavioural half is directional
    and only the cost decomposition is a number.
    """
    arms = {}
    for name, ratio in (("n32_r1", 1), ("n32_r4", 4)):
        data = json.load(open(os.path.join(ROOT, "benchmarks", f"utd_sac_{name}_5m.json"),
                              encoding="utf-8"))
        clock = UTD_5M_WALL_CLOCK[name]
        arms[name] = {
            "num_envs": 32, "utd_ratio": ratio, "seed": 7, "physics_preset": data["physics_preset"],
            "device": data.get("device"), "torch_threads": data.get("torch_threads"),
            "scored_in_version": data.get("scored_in_version"),
            "checkpoint_run": data["models"]["s1000000"]["checkpoint"].split("/")[1],
            "updates_per_environment_step": round(ratio / 32.0, 4),
            "wall_clock_seconds": clock["seconds"], "wall_clock_window": clock["window"],
            "seconds_per_1m": round(clock["seconds"] / 5.0, 1),
            "scoring_seconds": clock["scoring_seconds"],
            "by_budget": {
                budget: {k: model[k] for k in ("mean", "std", "reached_target_pct",
                                               "mean_min_target_distance", "mean_x_velocity",
                                               "standing_at_end_pct", "falls_per_episode")}
                for budget, model in sorted(data["models"].items(), key=lambda kv: int(kv[0][1:]))
            },
        }
    r1, r4 = arms["n32_r1"], arms["n32_r4"]
    cost_ratio = r4["wall_clock_seconds"] / r1["wall_clock_seconds"]
    update_seconds = (r4["wall_clock_seconds"] - r1["wall_clock_seconds"]) / 3.0
    collection_seconds = r1["wall_clock_seconds"] - update_seconds
    at5 = lambda arm: arm["by_budget"]["s5000000"]
    return {
        "protocol": ("SAC 5M, seed 7, task_phase=target, reset_mode=mixed, "
                     "target_forward_velocity=1.2, num_envs=32, one arm at --utd-ratio 1 and one at "
                     "4, back to back in one window, checkpointed every 1M; each scored with "
                     "eval_phase1.py --num-episodes 20 --seed 11 in v9"),
        "arms": arms,
        "ratio_4_over_ratio_1_seconds": round(cost_ratio, 3),
        "derived_from_the_two_measurements": {
            "update_seconds_at_ratio_1": round(update_seconds, 1),
            "collection_seconds": round(collection_seconds, 1),
            "update_share_of_ratio_1_clock_pct": round(100 * update_seconds
                                                       / r1["wall_clock_seconds"], 1),
            "equations": ("C + U = 2399 and C + 4U = 6494, where U is the cost of the updates one "
                          "ratio-1 run performs and C the collection it shares with the ratio-4 run"),
        },
        "dose_matched_against_the_committed_8_env_arm": {
            "n8_ratio1_seconds": 8992,
            "n8_ratio1_source": "benchmarks/physics_presets_screen5m_paired.json (chain18, 2026-10-05)",
            "updates_per_environment_step": round(1 / 8.0, 4),
            "speedup_of_n32_ratio4": round(8992 / r4["wall_clock_seconds"], 3),
            "caveat": ("a different window on a different night: the same world at 8 envs ran at "
                       "1,798.4 s per 1M there and 1,877.2 s per 1M in chain23, so this ratio carries "
                       "the per-window spread every other pair in the README does"),
        },
        "reach_pct_at_5m": {name: at5(arm)["reached_target_pct"] for name, arm in arms.items()},
        "mean_at_5m": {name: at5(arm)["mean"] for name, arm in arms.items()},
        "note": ("the two columns rank the arms in opposite directions, which is the point: the mean "
                 "return is mostly posture shaping and the reach column is the task's own criterion"),
    }, out


UTD_SEED8_WALL_CLOCK = {
    # chain26's launcher log: the same two doses re-run at seed 8, back to back in one window, so the
    # cost ratio of the second draw is comparable to the first draw's and to nothing else.
    "n32_seed8_r1": {"seconds": 2529, "scoring_seconds": 86,
                     "window": "2026-10-06 23:43:58 -> 2026-10-07 00:26:07"},
    "n32_seed8_r4": {"seconds": 6824, "scoring_seconds": 86,
                     "window": "2026-10-07 00:27:33 -> 02:21:17"},
}


def utd_dose_draws(out="benchmarks/utd_dose_draws.json"):
    """The dose pair drawn twice, which is what turns 'directional' into a claim about the task.

    Draw 1 said the score columns disagreed with each other and named itself one draw per arm. Draw 2
    (seed 8, same window discipline) reproduces the cost arithmetic almost exactly - **2.698x for four
    times the updates** against draw 1's 2.707x, 0.3% apart - and it also reproduces the behavioural
    direction: the higher dose reaches the target more often, in **6 of the 8 cells that are not ties**
    and at 5M in both draws (20% and 15% against 5% and 5%, i.e. 14 of 40 episodes against 4 of 40).

    The mean return still ranks the arms the other way on average (32,158 against 29,266 over the two
    5M draws), which is the point rather than a complication: after two draws the return column has not
    moved with the task's own criterion, and this section has now measured that twice on different
    axes - across physics worlds and across optimisation doses.
    """
    draws = {}
    for tag, files, clocks in (
            ("seed7", {1: "utd_sac_n32_r1_5m.json", 4: "utd_sac_n32_r4_5m.json"}, UTD_5M_WALL_CLOCK),
            ("seed8", {1: "utd_sac_n32_seed8_r1_5m.json", 4: "utd_sac_n32_seed8_r4_5m.json"},
             UTD_SEED8_WALL_CLOCK)):
        arms = {}
        for ratio, fname in files.items():
            data = json.load(open(os.path.join(ROOT, "benchmarks", fname), encoding="utf-8"))
            key = f"n32_r{ratio}" if tag == "seed7" else f"n32_seed8_r{ratio}"
            clock = clocks[key]
            arms[ratio] = {
                "seed": 7 if tag == "seed7" else 8, "utd_ratio": ratio, "num_envs": 32,
                "updates_per_environment_step": round(ratio / 32.0, 4),
                "device": data.get("device"), "torch_threads": data.get("torch_threads"),
                "scored_in_version": data.get("scored_in_version"),
                "physics_preset": data.get("physics_preset"),
                "checkpoint_run": data["models"]["s1000000"]["checkpoint"].split("/")[1],
                "wall_clock_seconds": clock["seconds"], "wall_clock_window": clock["window"],
                "seconds_per_1m": round(clock["seconds"] / 5.0, 1),
                "scoring_seconds": clock["scoring_seconds"],
                "by_budget": {
                    budget: {k: model[k] for k in ("mean", "std", "reached_target_pct",
                                                   "mean_min_target_distance", "mean_x_velocity",
                                                   "standing_at_end_pct", "falls_per_episode")}
                    for budget, model in sorted(data["models"].items(),
                                                key=lambda kv: int(kv[0][1:]))
                },
            }
        faster = arms[4]["wall_clock_seconds"] / arms[1]["wall_clock_seconds"]
        draws[tag] = {"arms": arms, "cost_ratio_of_four_times_the_updates": round(faster, 3)}

    budgets = [f"s{m}000000" for m in range(1, 6)]
    reach = {ratio: {tag: [draws[tag]["arms"][ratio]["by_budget"][b]["reached_target_pct"]
                           for b in budgets] for tag in draws} for ratio in (1, 4)}
    won = lost = tied = 0
    for tag in draws:
        for i in range(5):
            hi, lo = reach[4][tag][i], reach[1][tag][i]
            if hi > lo:
                won += 1
            elif hi < lo:
                lost += 1
            else:
                tied += 1
    at5 = {ratio: [draws[tag]["arms"][ratio]["by_budget"]["s5000000"] for tag in draws]
           for ratio in (1, 4)}
    return {
        "protocol": ("SAC 5M, num_envs=32, task_phase=target, reset_mode=mixed, "
                     "target_forward_velocity=1.2, in the published world v9, one arm at "
                     "--utd-ratio 1 and one at 4 for each seed (7 in chain24, 8 in chain26), each "
                     "arm scored with eval_phase1.py --num-episodes 20 --seed 11; the two arms of a "
                     "draw ran back to back in one window"),
        "draws": draws,
        "cost_ratios_per_draw": {tag: draws[tag]["cost_ratio_of_four_times_the_updates"]
                                 for tag in draws},
        "reproduced_to_pct": round(100 * abs(draws["seed7"]["cost_ratio_of_four_times_the_updates"]
                                             - draws["seed8"]["cost_ratio_of_four_times_the_updates"])
                                   / draws["seed7"]["cost_ratio_of_four_times_the_updates"], 1),
        "update_share_of_ratio_1_clock_pct": {
            tag: round(100 * ((draws[tag]["arms"][4]["wall_clock_seconds"]
                               - draws[tag]["arms"][1]["wall_clock_seconds"]) / 3.0)
                       / draws[tag]["arms"][1]["wall_clock_seconds"], 1) for tag in draws},
        "reach_pct_by_budget": reach,
        "dose4_cells": {"ahead": won, "behind": lost, "tied": tied},
        "mean_over_upper_budgets_pct": {
            ratio: round(sum(sum(reach[ratio][tag][2:]) for tag in draws) / 6.0, 2)
            for ratio in (1, 4)},
        "at_5m": {ratio: {"means": [m["mean"] for m in at5[ratio]],
                          "reaches_pct": [m["reached_target_pct"] for m in at5[ratio]],
                          "closest_m": [m["mean_min_target_distance"] for m in at5[ratio]],
                          "reached_episodes_of_40": int(sum(m["reached_target_pct"]
                                                            for m in at5[ratio]) * 40 / 100)}
                  for ratio in (1, 4)},
        "note": ("two draws per dose: the cost sublinearity and the reach advantage both replicate, "
                 "while the mean return ranks the arms the other way on average - the return column is "
                 "posture and the reach column is the task, now measured on two different axes"),
    }, out


TRAINER_PAIR_WALL_CLOCK = {
    # Recorded from the launcher logs of chains 19 and 20, which ran back to back on an idle
    # machine: PPO v9 at 01:42:06 -> 01:46:48, PPO fast at 01:46:48 -> 01:50:52, SAC at n=32 at
    # 01:57:13 -> 02:07:07. `collected_steps` is what the trainer actually reached: PPO checkpoints
    # only at rollout boundaries, so its 1M budget ends at 983,040 steps (15 rollouts of 2048x32).
    "ppo_v9": {"seconds": 282, "collected_steps": 983040, "checkpoint":
               "checkpoints/screen_ppo_v9_1m/ppo_ckpt_983040.pt",
               "scored_at": 983040},
    "ppo_fast": {"seconds": 244, "collected_steps": 983040, "checkpoint":
                 "checkpoints/screen_ppo_fast_1m/ppo_ckpt_983040.pt",
                 "scored_at": 983040},
    "sac_v9": {"seconds": 594, "collected_steps": 1000000, "checkpoint":
               "checkpoints/screen_sac_v9_1m_n32/sac_actor_1000000.pt",
               "scored_at": 1000000},
}


def trainer_pair_ppo_sac(out="benchmarks/trainer_pair_ppo_sac.json"):
    """Is PPO a faster way to train this task than SAC? Both halves of the question, measured.

    PPO is faster per environment step and it is not closer to the behaviour, and the two facts have
    to be said in the same breath or the first one reads as a recommendation. Same config for both
    trainers (32 envs, target phase, mixed resets, seed 7, cuda), scored at the published protocol.
    The earlier budget pair is kept as well, because PPO's second-best checkpoint is within 5% of
    SAC's 500k one and the ordering is the same at both.
    """
    arms = {}
    for key, spec in TRAINER_PAIR_WALL_CLOCK.items():
        preset = "fast" if key.endswith("fast") else "v9"
        budget = spec["scored_at"]
        if key.startswith("sac"):
            data = json.load(open(os.path.join(ROOT, "benchmarks", "trainer_pair_sac_n32.json"),
                                  encoding="utf-8"))
            model = data["models"]["sac1m"]
        else:
            data = json.load(open(os.path.join(ROOT, "benchmarks",
                                               f"ppo_screen_{preset}_{budget}.json"),
                                  encoding="utf-8"))
            model = data["models"][f"ppo{budget}"]
        arms[key] = {
            "seconds": spec["seconds"], "collected_steps": spec["collected_steps"],
            "env_steps_per_second": round(spec["collected_steps"] / spec["seconds"], 1),
            "seconds_per_1m_env_steps": round(spec["seconds"] * 1e6 / spec["collected_steps"], 1),
            "checkpoint": spec["checkpoint"], "algo": model["algo"],
            "device": data.get("device"), "torch_threads": data.get("torch_threads"),
            "scored_in_version": data.get("scored_in_version"),
            "mean": model["mean"], "std": model["std"],
            "reached_target_pct": model["reached_target_pct"],
            "mean_min_target_distance": model["mean_min_target_distance"],
            "mean_x_velocity": model["mean_x_velocity"],
            "standing_at_end_pct": model["standing_at_end_pct"],
            "falls_per_episode": model["falls_per_episode"],
        }
    ppo, sac = arms["ppo_v9"], arms["sac_v9"]
    # The earlier budget of the same night, kept in this artifact so "the ordering is not an artefact
    # of one checkpoint" is a checkable statement rather than a aside in prose.
    ppo_data = json.load(open(os.path.join(ROOT, "benchmarks", "ppo_screen_v9_524288.json"),
                              encoding="utf-8"))["models"]["ppo524288"]
    sac_data = json.load(open(os.path.join(ROOT, "benchmarks", "trainer_pair_sac_n32.json"),
                              encoding="utf-8"))["models"]["sac500k"]
    earlier = {
        "ppo_524288": {"steps": ppo_data["global_step"], "mean": ppo_data["mean"],
                       "mean_min_target_distance": ppo_data["mean_min_target_distance"],
                       "falls_per_episode": ppo_data["falls_per_episode"],
                       "reached_target_pct": ppo_data["reached_target_pct"],
                       "checkpoint": "checkpoints/screen_ppo_v9_1m/ppo_ckpt_524288.pt"},
        "sac_500000": {"steps": sac_data["global_step"], "mean": sac_data["mean"],
                       "mean_min_target_distance": sac_data["mean_min_target_distance"],
                       "falls_per_episode": sac_data["falls_per_episode"],
                       "reached_target_pct": sac_data["reached_target_pct"],
                       "checkpoint": "checkpoints/screen_sac_v9_1m_n32/sac_actor_500000.pt"},
    }
    return {
        "protocol": ("both trainers at num_envs=32, task_phase=target, reset_mode=mixed, seed 7, "
                     "target_forward_velocity=1.2, cuda, ~1M env-step budget; scored with "
                     "eval_phase1.py --num-episodes 20 --seed 11 in the world each was trained in"),
        "arms": arms,
        "earlier_budget": earlier,
        "ppo_steps_per_second_over_sac": round(ppo["env_steps_per_second"]
                                               / sac["env_steps_per_second"], 3),
        "note": ("PPO reaches 983,040 steps, not 1,000,000: it checkpoints at rollout boundaries "
                 "(15 rollouts of 2048x32), so the two budgets are within 1.7% of each other and "
                 "the rate is quoted per collected step, not per nominal step"),
    }, out


DREAMER_PRESET_WALL_CLOCK = {
    # Recorded from chain21's launcher log: the two arms back to back in one window, same seed, same
    # budget, same captured-update configuration (`[DREAMER] update: captured CUDA graph` in both).
    "v9": {"seconds": 783, "window": "2026-10-06 10:54:56 -> 11:07:59"},
    "fast": {"seconds": 690, "window": "2026-10-06 11:07:59 -> 11:19:29"},
}


def physics_preset_dreamer_pair(out="benchmarks/physics_presets_dreamer_pair.json"):
    """The preset on the trainer whose wall clock is the update, not the physics.

    SAC and PPO gained 1.15-1.23x. Dreamer is the hard case for a cheaper environment: 78.6% of its
    iteration is the captured update, so the arithmetic of the gain is fixed by that share - divide
    the collection part by the measured env-step ratio, leave the update alone - and the run either
    lands on the prediction or it does not. It lands on it, which is the useful result: the preset's
    value is predictable from a loop split rather than something to be discovered per trainer.
    """
    split = json.load(open(os.path.join(ROOT, "benchmarks", "dreamer_loop_split.json"),
                           encoding="utf-8"))
    preset = json.load(open(os.path.join(ROOT, "benchmarks", "physics_presets.json"),
                            encoding="utf-8"))
    env_ratio = preset["throughput_ratio"]["fast"][f"env_step_n{split['num_envs']}_vs_v9"] \
        if f"env_step_n{split['num_envs']}_vs_v9" in preset["throughput_ratio"]["fast"] else None
    if env_ratio is None:
        env_ratio = preset["throughput_ratio"]["fast"]["env_step_n8_vs_v9"]
    update_share = split["update_share_pct"] / 100.0
    predicted = 1.0 / (update_share + (1.0 - update_share) / env_ratio)
    measured = DREAMER_PRESET_WALL_CLOCK["v9"]["seconds"] / DREAMER_PRESET_WALL_CLOCK["fast"]["seconds"]
    return {
        "protocol": ("DreamerV3 build, 250k env steps, num_envs=8, seed 7, task_phase=target, "
                     "reset_mode=mixed, captured CUDA update in both arms; one window, arms back to "
                     "back, v9 first"),
        "arms": {k: dict(v, steps=250000,
                         env_steps_per_second=round(250000 / v["seconds"], 1))
                 for k, v in DREAMER_PRESET_WALL_CLOCK.items()},
        "measured_speedup_fast_vs_v9": round(measured, 3),
        "predicted_from_loop_split": {
            "value": round(predicted, 3),
            "update_share_pct": split["update_share_pct"],
            "env_step_ratio_used": env_ratio,
            "formula": "1 / (update_share + (1 - update_share) / env_step_ratio)",
            "source": "benchmarks/dreamer_loop_split.json and benchmarks/physics_presets.json",
        },
        "prediction_error_pct": round(100.0 * (measured - predicted) / predicted, 1),
    }, out


SAC_1M_TRIPLE = {
    # Wall clocks from chain22_evidence.log, three arms back to back in one window, and the scoring
    # spans of the same window. The window matters: `v9` took 3,044 s here against 1,345 s in the
    # window that produced the committed 1M pair, which is the machine's state, not the worlds.
    "v9": {"train_seconds": 3044, "eval_seconds": 28, "window": "2026-10-06 12:55:56 -> 13:46:40"},
    "euler": {"train_seconds": 2630, "eval_seconds": 21,
              "window": "2026-10-06 13:47:08 -> 14:30:58"},
    "fast": {"train_seconds": 1960, "eval_seconds": 14,
             "window": "2026-10-06 14:31:19 -> 15:03:59"},
}


def physics_presets_sac_triple(out="benchmarks/physics_presets_sac1m_triple.json"):
    """The three physics worlds trained end to end at 1M, the case the step ratio cannot speak for.

    `euler` is the interesting arm: it drifts least from the published world (0.227 m of torso
    height over a shared episode against 0.301 m) and until this run had never been trained in. The
    question is whether being closer to v9 costs it the speed, and the answer is that it does lose
    some - but the more arresting column is the last one. At 1M the three worlds do not tie, and
    the ordering follows the price of the step, which is precisely why the second draw at 5M exists.
    """
    arms = {}
    for preset, spec in SAC_1M_TRIPLE.items():
        data = json.load(open(os.path.join(ROOT, "benchmarks",
                                           f"physics_presets_sac1m_{preset}.json"),
                              encoding="utf-8"))
        model = data["models"][f"sac1m_{preset}"]
        arms[preset] = dict(spec, steps=1000000,
                            env_steps_per_second=round(1000000 / spec["train_seconds"], 1),
                            scored_in_version=data.get("scored_in_version"),
                            device=data.get("device"), torch_threads=data.get("torch_threads"),
                            mean=model["mean"], std=model["std"],
                            reached_target_pct=model["reached_target_pct"],
                            mean_min_target_distance=model["mean_min_target_distance"],
                            falls_per_episode=model["falls_per_episode"],
                            standing_at_end_pct=model["standing_at_end_pct"])
    base = arms["v9"]["train_seconds"]
    return {
        "protocol": ("SAC 1M, seed 7, num_envs=8, task_phase=target, reset_mode=mixed, "
                     "target_forward_velocity=1.2, one arm per physics preset in the same window "
                     "(v9, euler, fast), each scored with eval_phase1.py --num-episodes 20 --seed 11 "
                     "in the world it trained in"),
        "arms": arms,
        "train_seconds_over_v9": {k: round(base / v["train_seconds"], 3) for k, v in arms.items()},
        "step_ratio_measured_in_its_own_window": {
            k: json.load(open(os.path.join(ROOT, "benchmarks", "physics_presets.json"),
                               encoding="utf-8"))["throughput_ratio"][k]["env_step_n8_vs_v9"]
            for k in ("euler", "fast")},
        "note": ("the step ratio and the end-to-end ratio are different measurements of different "
                 "things: the first is one vector step with the learner switched off, the second is "
                 "a whole training run with it on, and the gap between 2.7x and 1.55x is the part "
                 "of SAC's clock the preset cannot touch"),
    }, out


PPO_10M_WALL_CLOCK = {
    # Recorded from chain25's launcher log. PPO checkpoints at rollout boundaries and writes no
    # elapsed time into them, and `train_walker.py` logs nothing to mlflow, so the run's whole cost is
    # this one span - which is also why the comparison below is made between whole runs and never
    # "at what second" a checkpoint was good.
    "seconds": 1594, "scoring_seconds": 130,
    "window": "2026-10-06 23:03:01 -> 23:29:35",
}


def ppo_10m_against_sac(out="benchmarks/ppo_10m_vs_sac.json"):
    """Ten times the budget for PPO, and the clock of every SAC arm it should be judged against.

    The committed pair said PPO is the fastest loop and the worst policy at ~1M steps
    (`benchmarks/trainer_pair_ppo_sac.json`), and left the obvious objection on the table: 1M is early
    for on-policy. This is the same recipe at 10M, and the objection does not survive it - PPO stands
    up, walks, and reaches the target in 3 of 20 episodes at 5,046,272 steps.

    What the wall clock says is the part worth acting on: **9,961,472 steps in 1,594 s = 6,249.4
    env-steps/s**, which is 3.0x the rate of SAC at `num_envs=32` ratio 1 measured over the same 5M
    (479.8 s per 1M) and 11.2x the rate of the 8-env screens (1,798.4 s per 1M). Against the SAC arm of
    the committed 1M pair it is 3.7x, and the gap between 3.0 and 3.7 is the same trap this repository
    has been caught in before: that pair's SAC ran 1M in 594 s, so its rate divides the run's fixed
    costs by a sixth of the steps. The whole PPO curve, including its best reach column, cost less
    clock than a single SAC 5M arm in this world.

    Two honest limits travel with it. The rate is higher than the committed pair's 3,486.0/s because
    that arm paid its fixed startup over 983,040 steps instead of 9.96M - a short run's rate divides
    the same fixed cost by fewer steps. And this is one draw: the reach column inside this very run
    oscillates 0-15% across ten checkpoints, which is the band the same evening measured at ~20k of
    mean return in SAC, so the *shape* (PPO comes up late and reaches occasionally) is the finding and
    the individual cells are not.
    """
    ppo = json.load(open(os.path.join(ROOT, "benchmarks", "ppo_learning_curve_10m_v9.json"),
                         encoding="utf-8"))
    budgets = sorted(ppo["models"], key=lambda k: int(k[1:]))
    total = int(budgets[-1][1:])
    clock = PPO_10M_WALL_CLOCK
    pair = json.load(open(os.path.join(ROOT, "benchmarks", "trainer_pair_ppo_sac.json"),
                          encoding="utf-8"))
    dose = json.load(open(os.path.join(ROOT, "benchmarks", "utd_dose_pair.json"), encoding="utf-8"))
    draws = json.load(open(os.path.join(ROOT, "benchmarks",
                                        "physics_presets_screen5m_draws.json"), encoding="utf-8"))
    five = lambda arm: arm["by_budget"]["s5000000"]
    sac_rows = {
        "sac_n32_r1": {"seconds": dose["arms"]["n32_r1"]["wall_clock_seconds"],
                       "steps": 5000000,
                       "mean_at_5m": five(dose["arms"]["n32_r1"])["mean"],
                       "reached_pct_at_5m": five(dose["arms"]["n32_r1"])["reached_target_pct"]},
        "sac_n32_r4": {"seconds": dose["arms"]["n32_r4"]["wall_clock_seconds"],
                       "steps": 5000000,
                       "mean_at_5m": five(dose["arms"]["n32_r4"])["mean"],
                       "reached_pct_at_5m": five(dose["arms"]["n32_r4"])["reached_target_pct"]},
        "sac_n8_r1_seed7": {"seconds": draws["draws"]["seed7"]["arms"]["v9"]["wall_clock_seconds"],
                            "steps": 5000000,
                            "mean_at_5m": five(draws["draws"]["seed7"]["arms"]["v9"])["mean"],
                            "reached_pct_at_5m": five(draws["draws"]["seed7"]["arms"]["v9"])["reached_target_pct"]},
        "sac_n8_r1_seed8": {"seconds": draws["draws"]["seed8"]["arms"]["v9"]["wall_clock_seconds"],
                            "steps": 5000000,
                            "mean_at_5m": five(draws["draws"]["seed8"]["arms"]["v9"])["mean"],
                            "reached_pct_at_5m": five(draws["draws"]["seed8"]["arms"]["v9"])["reached_target_pct"]},
    }
    for row in sac_rows.values():
        row["seconds_per_1m"] = round(row["seconds"] / (row["steps"] / 1e6), 1)
    ppo_per_1m = round(clock["seconds"] / (total / 1e6), 1)
    ppo_rate = round(total / clock["seconds"], 1)
    best = max(ppo["models"].values(), key=lambda m: m["reached_target_pct"])
    return {
        "protocol": ("PPO, seed 7, num_envs=32, task_phase=target, reset_mode=mixed, "
                     "target_forward_velocity=1.2, 10M requested and 9,961,472 collected (the trainer "
                     "checkpoints at rollout boundaries of 2048x32 steps), scored by eval_phase1.py "
                     "--num-episodes 20 --seed 11 in v9; SAC rows are the arms in "
                     "benchmarks/utd_dose_pair.json and physics_presets_screen5m_draws.json"),
        "ppo_10m": {
            "wall_clock_seconds": clock["seconds"], "wall_clock_window": clock["window"],
            "scoring_seconds": clock["scoring_seconds"], "collected_steps": total,
            "env_steps_per_second": ppo_rate, "seconds_per_1m": ppo_per_1m,
            "device": ppo.get("device"), "torch_threads": ppo.get("torch_threads"),
            "scored_in_version": ppo.get("scored_in_version"),
            "best_reach": {"step": int(max(ppo["models"], key=lambda k: ppo["models"][k]["reached_target_pct"])[1:]),
                           "reached_pct_at_step": best["reached_target_pct"], "mean": best["mean"],
                           "mean_min_target_distance": best["mean_min_target_distance"],
                           "standing_at_end_pct": best["standing_at_end_pct"],
                           "falls_per_episode": best["falls_per_episode"]},
            "curve": {b: {k: ppo["models"][b][k] for k in ("mean", "std", "reached_target_pct",
                                                           "mean_min_target_distance",
                                                           "mean_x_velocity", "standing_at_end_pct",
                                                           "falls_per_episode")} for b in budgets},
        },
        "ppo_1m_committed_arm": {
            "seconds": pair["arms"]["ppo_v9"]["seconds"],
            "collected_steps": pair["arms"]["ppo_v9"]["collected_steps"],
            "mean": pair["arms"]["ppo_v9"]["mean"],
            "reached_pct": pair["arms"]["ppo_v9"]["reached_target_pct"],
            "mean_min_target_distance": pair["arms"]["ppo_v9"]["mean_min_target_distance"],
            "env_steps_per_second": pair["arms"]["ppo_v9"]["env_steps_per_second"]
            if "env_steps_per_second" in pair["arms"]["ppo_v9"] else None,
        },
        "sac_arms_same_task_same_world": sac_rows,
        "rate_ratios": {
            "ppo_over_sac_n32_r1": round(sac_rows["sac_n32_r1"]["seconds_per_1m"] / ppo_per_1m, 1),
            "ppo_over_sac_n32_r4": round(sac_rows["sac_n32_r4"]["seconds_per_1m"] / ppo_per_1m, 1),
            "ppo_over_sac_n8_r1": round(sac_rows["sac_n8_r1_seed7"]["seconds_per_1m"] / ppo_per_1m, 1),
            "ppo_over_sac_1m_pair_arm": round(ppo_rate
                                              / pair["arms"]["sac_v9"]["env_steps_per_second"], 1),
            "the_last_two_disagree_because": ("the pair's SAC arm is 1M over 594 s, so its per-second "
                                              "rate divides fixed startup by one sixth of the steps of "
                                              "the 5M arms; only the same-budget ratios are usable"),
        },
        "note": ("PPO at 10M stands, walks and reaches, and it does so on a clock that is a fraction of "
                 "any SAC arm here - but the reach column inside this one run swings 0 to 15% across "
                 "ten checkpoints, so read the shape and not the cells"),
    }, out


EULER_5M_WALL_CLOCK = {
    # chain27 (seed 7) and chain28 (seed 8), each a window of its own so the euler-vs-v9 ratio is
    # computed inside a draw and never across nights. Re-checked against the committed logs.
    "euler_seed7": {"seconds": 7860, "scoring_seconds": 58,
                    "window": "2026-10-07 02:35:53 -> 04:46:53"},
    "euler_seed8": {"seconds": 7505, "scoring_seconds": 56,
                    "window": "2026-10-07 04:52:41 -> 06:57:46"},
}


def euler_screen_5m_draws(out="benchmarks/physics_presets_screen5m_euler_draws.json"):
    """`euler` at 5M twice, because the first time it reached nothing and one draw says nothing.

    Draw 1 (seed 7) flat-lined: mean 3,150.93 -> -76.12 across five checkpoints, forward speed ~0,
    0.10 falls per episode, and the target reached in 0 of 100 scored episodes - the lying-still
    optimum PPO sat in at 1M. Draw 2 (seed 8), same world and same budget, climbed to
    **14,409.67 at 4M and reached in 10% of episodes at 5M**. So the first result was a draw, not a
    property of the integrator - which is the inference the paired-seed design exists to license, and
    it was bought for 2 h 05 m of machine time rather than argued.

    The cost half replicates on its own terms: 7,860 s against the published world's 8,992 s at seed 7
    (**1.144x**) and 7,505 s against 9,386 s at seed 8 (**1.251x**), two independent windows bracketing
    the 1.157x the 1M triple measured. And the behavioural comparison to `v9` closes: euler's two-draw
    reach record at 5M is 0% and 10%, v9's is 10% and 0% - indistinguishable at this sample size, while
    the `fast` world led both of its draws. The world that drifts least (0.227 m of torso height over a
    shared episode) tracks the published one on the criterion, which is what a screening preset should
    do; the world that prunes self-collision is where the divergence lives.
    """
    draws = {}
    for tag, fname, seed in (("seed7", "physics_presets_screen5m_euler.json", 7),
                             ("seed8", "physics_presets_screen5m_seed8_euler.json", 8)):
        data = json.load(open(os.path.join(ROOT, "benchmarks", fname), encoding="utf-8"))
        clock = EULER_5M_WALL_CLOCK[f"euler_{tag}"]
        v9 = json.load(open(os.path.join(ROOT, "benchmarks",
                                         "physics_presets_screen5m_draws.json"),
                            encoding="utf-8"))["draws"][tag]["arms"]["v9"]
        faster = v9["wall_clock_seconds"] / clock["seconds"]
        draws[tag] = {
            "seed": seed, "physics_preset": data["physics_preset"],
            "scored_in_version": data.get("scored_in_version"),
            "device": data.get("device"), "torch_threads": data.get("torch_threads"),
            "checkpoint_run": data["models"]["s1000000"]["checkpoint"].split("/")[1],
            "wall_clock_seconds": clock["seconds"], "wall_clock_window": clock["window"],
            "seconds_per_1m": round(clock["seconds"] / 5.0, 1),
            "scoring_seconds": clock["scoring_seconds"],
            "v9_same_draw_seconds": v9["wall_clock_seconds"],
            "v9_same_draw_seconds_per_1m": v9["seconds_per_1m"],
            "speedup_euler_over_v9_in_this_window": round(faster, 3),
            "by_budget": {
                budget: {k: model[k] for k in ("mean", "std", "reached_target_pct",
                                               "mean_min_target_distance", "mean_x_velocity",
                                               "standing_at_end_pct", "falls_per_episode")}
                for budget, model in sorted(data["models"].items(),
                                            key=lambda kv: int(kv[0][1:]))
            },
        }
    at5 = lambda tag: draws[tag]["by_budget"]["s5000000"]
    v9_draws = json.load(open(os.path.join(ROOT, "benchmarks",
                                           "physics_presets_screen5m_draws.json"),
                              encoding="utf-8"))["draws"]
    return {
        "protocol": ("SAC 5M, num_envs=8, task_phase=target, reset_mode=mixed, "
                     "target_forward_velocity=1.2, trained in `euler` once per seed (7 in chain27, "
                     "8 in chain28), each draw scored with eval_phase1.py --num-episodes 20 --seed 11 "
                     "in the world it trained in, against the `v9` arm of the same seed from "
                     "benchmarks/physics_presets_screen5m_draws.json"),
        "draws": draws,
        "speedups_euler_over_v9": {tag: draws[tag]["speedup_euler_over_v9_in_this_window"]
                                   for tag in draws},
        "at_1m_triple_for_reference": 1.157,
        "reached_target_pct_at_5m": {
            "euler": [at5(t)["reached_target_pct"] for t in draws],
            "v9": [v9_draws[t]["arms"]["v9"]["by_budget"]["s5000000"]["reached_target_pct"]
                   for t in draws]},
        "mean_at_5m": {"euler": [at5(t)["mean"] for t in draws]},
        "scored_episodes_reaching": {"euler": int(sum(at5(t)["reached_target_pct"]
                                                      for t in draws) * 40 / 100),
                                     "of_episodes": 40},
        "note": ("draw 1 reached in 0 of 100 episodes and draw 2 in 2 of 20: the flat line was the run. "
                 "Both worlds' two-draw reach records are {0%, 10%} and {10%, 0%}, which this sample "
                 "size cannot separate"),
    }, out


PPO_10M_SEED8_WALL_CLOCK = {
    # chain29's launcher log. The same recipe, in the same world, took 2.28x the clock of chain25's
    # window - which is why every cross-algorithm rate in this section is now labelled with the window
    # it came from.
    "seconds": 3636, "scoring_seconds": 132,
    "window": "2026-10-07 07:12:18 -> 08:12:54",
}


def ppo_10m_draws(out="benchmarks/ppo_10m_draws.json"):
    """PPO at 10M twice: the behaviour replicates, the clock does not.

    Draw 1 (seed 7, chain25) stood, walked and reached: mean from **-813.28** to **43,439.25**, best
    reach **15%** at 5,046,272. Draw 2 (seed 8, chain29) repeats the shape - mean 4,397.74 to
    **34,707.81** with its own best reach of **25%** at 7,012,352 - so "PPO needed ten times the budget,
    it did not need a different algorithm" survives two draws, and the level column swings between them
    exactly as it does for SAC.

    The clock is the finding that did not survive. The identical recipe cost **1,594 s** in one window
    and **3,636 s** in the next: **2.28x**, on a laptop that was on AC power in both. That refutes the
    sentence published from draw 1 - that the whole PPO curve costs less clock than one SAC 5M arm in
    this world - because 3,636 s is *more* than SAC's ratio-1 arms (2,399 s and 2,529 s) and less than
    its ratio-4 arms (6,494 s and 6,824 s). The honest statement is "PPO at 10M and SAC at 5M cost about
    the same, straddling SAC's dose choice", and the rate ratios quoted from draw 1 (3.0x, 11.2x) are
    cross-window numbers, not a property of the two trainers.

    Nothing here can say *why* the two windows differ by 2.28x: chain28's two-hour GPU run ended four
    minutes before chain29 started, so sustained thermal throttling is the obvious candidate and an
    idle-gap replication is the obvious test. Both are recorded as open rather than resolved, and the
    numbers are labelled per window so a later reader cannot quietly reuse one as a constant.
    """
    draws = {}
    for tag, fname, clock in (("seed7", "ppo_learning_curve_10m_v9.json", PPO_10M_WALL_CLOCK),
                              ("seed8", "ppo_learning_curve_10m_v9_seed8.json",
                               PPO_10M_SEED8_WALL_CLOCK)):
        data = json.load(open(os.path.join(ROOT, "benchmarks", fname), encoding="utf-8"))
        budgets = sorted(data["models"], key=lambda k: int(k[1:]))
        total = int(budgets[-1][1:])
        best_step = max(budgets, key=lambda b: data["models"][b]["reached_target_pct"])
        peak_step = max(budgets, key=lambda b: data["models"][b]["mean"])
        draws[tag] = {
            "seed": 7 if tag == "seed7" else 8,
            "collected_steps": total, "wall_clock_seconds": clock["seconds"],
            "wall_clock_window": clock["window"], "scoring_seconds": clock["scoring_seconds"],
            "env_steps_per_second": round(total / clock["seconds"], 1),
            "seconds_per_1m": round(clock["seconds"] / (total / 1e6), 1),
            "device": data.get("device"), "torch_threads": data.get("torch_threads"),
            "scored_in_version": data.get("scored_in_version"),
            "physics_preset": data.get("physics_preset"),
            "checkpoint_run": data["models"][budgets[0]]["checkpoint"].split("/")[1],
            "best_reach": {"step": int(best_step[1:]),
                           "reached_pct": data["models"][best_step]["reached_target_pct"],
                           "mean": data["models"][best_step]["mean"],
                           "mean_min_target_distance":
                               data["models"][best_step]["mean_min_target_distance"]},
            "peak_mean": {"step": int(peak_step[1:]),
                          "mean": data["models"][peak_step]["mean"],
                          "falls_per_episode": data["models"][peak_step]["falls_per_episode"]},
            "final": {"step": total, "mean": data["models"][budgets[-1]]["mean"],
                      "reached_pct": data["models"][budgets[-1]]["reached_target_pct"]},
            "curve": {b: {k: data["models"][b][k] for k in ("mean", "reached_target_pct",
                                                            "mean_min_target_distance",
                                                            "mean_x_velocity",
                                                            "standing_at_end_pct",
                                                            "falls_per_episode")} for b in budgets},
        }
    d7, d8 = draws["seed7"], draws["seed8"]
    sac = json.load(open(os.path.join(ROOT, "benchmarks", "utd_dose_draws.json"), encoding="utf-8"))
    return {
        "protocol": ("PPO, num_envs=32, task_phase=target, reset_mode=mixed, "
                     "target_forward_velocity=1.2, 10M requested and 9,961,472 collected at rollout "
                     "boundaries, once per seed (7 in chain25, 8 in chain29), each checkpoint scored "
                     "with eval_phase1.py --num-episodes 20 --seed 11 in v9"),
        "draws": draws,
        "same_recipe_two_windows": {
            "seconds": [d7["wall_clock_seconds"], d8["wall_clock_seconds"]],
            "env_steps_per_second": [d7["env_steps_per_second"], d8["env_steps_per_second"]],
            "spread_ratio": round(d8["wall_clock_seconds"] / d7["wall_clock_seconds"], 3),
            "both_on_ac_power": True,
            "candidate_explanation_not_tested": ("chain28 ran 2 h of GPU immediately before chain29; "
                                                 "sustained throttling is the obvious suspect and an "
                                                 "idle-gap replication is the obvious test, neither "
                                                 "measured here"),
        },
        "against_sac_5m_arms_other_windows": {
            "sac_ratio1_seconds": [sac["draws"][t]["arms"]["1"]["wall_clock_seconds"]
                                   for t in ("seed7", "seed8")],
            "sac_ratio4_seconds": [sac["draws"][t]["arms"]["4"]["wall_clock_seconds"]
                                   for t in ("seed7", "seed8")],
            "ppo_seconds": [d7["wall_clock_seconds"], d8["wall_clock_seconds"]],
            "caveat": ("no SAC arm ran in either PPO window, so every ratio across the two algorithms "
                       "here is cross-window; draw 1 alone made PPO look 1.5x cheaper than SAC's "
                       "ratio-1 5M arm and draw 2 makes it 1.5x dearer"),
        },
        "behaviour_replicates": {
            "best_reach_pct": [d7["best_reach"]["reached_pct"], d8["best_reach"]["reached_pct"]],
            "peak_mean": [d7["peak_mean"]["mean"], d8["peak_mean"]["mean"]],
            "final_mean": [d7["final"]["mean"], d8["final"]["mean"]],
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


def mpc_target_rows(out="benchmarks/mpc_target_rows.json"):
    """A sampling MPC against the learned arms, on the same 20 episodes and the same step trace.

    The question this answers is the one the residual-learning idea rests on: does a controller with
    simulator access reach the marker on this ragdoll better than a policy that had to learn the
    dynamics? It does - but the same trace says it does it by falling toward the target, so the
    comparison splits rather than ranks, which is why both columns are in one artifact.

    Three cells: the small-search baseline, that baseline drawn a second time (the constraint solver is
    not bit-deterministic, so a single MPC run would be a claim waiting to be refuted the same way the
    PPO clock was), and a bigger-search cell that answers "the first budget was too small".
    """
    def cell(path):
        art = json.load(open(os.path.join(ROOT, "benchmarks", path), encoding="utf-8"))
        cfg, cost = art["config"], art["cost"]
        all_ep, reach, upright = (art["all_episodes"], art["reach_episodes"],
                                  art["upright_arrival_episodes"])
        return {
            "artifact": f"benchmarks/{path}", "config": cfg,
            "search": f"{cfg['samples']}x{cfg['iterations']} h{cfg['horizon']} replan{cfg['replan']}",
            "episodes": cfg["episodes"],
            "reach": reach["episodes"], "upright_arrivals": upright["episodes"],
            "reach_pct": round(100.0 * reach["episodes"] / cfg["episodes"], 1),
            "upright_pct": round(100.0 * upright["episodes"] / cfg["episodes"], 1),
            "share_closed_standing_all": all_ep["share_while_standing"],
            "share_closed_standing_upright": upright["share_while_standing"],
            "pct_steps_in_band": all_ep["mean_pct_steps_in_band"],
            "episodes_never_in_band": all_ep["episodes_never_in_band"],
            "max_longest_band_run_s": all_ep["max_longest_band_run_s"],
            "approach_after_first_fall": all_ep["closest_approach_after_first_fall"],
            "planning_s_per_episode": round(cost["plan_ms_mean"] * cost["plans_per_episode"] / 1e3, 1),
            "sim_steps_per_episode": cost["rollout_steps_per_episode"],
            "wall_clock_s": cost["wall_clock_s"],
        }

    rows = [cell("mpc_target_baseline.json"), cell("mpc_target_baseline_draw2.json"),
            cell("mpc_target_budget.json")]
    mech = json.load(open(os.path.join(ROOT, "benchmarks", "approach_mechanism.json"),
                          encoding="utf-8"))
    arr = json.load(open(os.path.join(ROOT, "benchmarks", "reach_upright_arrival.json"),
                         encoding="utf-8"))
    pool, gait = mech["pooled_v9_reach_episodes"], mech["pooled_v9_gait_reach_episodes"]
    best_learned = max((r for r in arr["rows"] if r["physics_preset"] == "v9"),
                       key=lambda r: r["reached_upright_count"])
    arm_a = [r for r in mech["rows"] if r["label"].startswith("stability 60, seed 7, best")]
    learned = {
        "artifact": "benchmarks/approach_mechanism.json, benchmarks/reach_upright_arrival.json",
        "best_learned_arm": {"label": best_learned["label"],
                             "upright_arrivals": best_learned["reached_upright_count"],
                             "upright_pct": best_learned["reached_upright_pct"],
                             "reach_pct": best_learned["reached_distance_only_pct"]},
        "published_pool_reach_episodes": {
            "episodes": pool["episodes"], "share_closed_standing": pool["share_while_standing"],
            "pct_steps_in_band": pool["mean_pct_steps_in_band"],
            "max_longest_band_run_s": pool["max_longest_band_run_s"]},
        "gait_pool_reach_episodes": {
            "episodes": gait["episodes"], "share_closed_standing": gait["share_while_standing"],
            "pct_steps_in_band": gait["mean_pct_steps_in_band"],
            "max_longest_band_run_s": gait["max_longest_band_run_s"]},
        "stability60_seed7_4m_share_closed_standing":
            arm_a[0]["reach_episodes"]["mean_share_while_standing"] if arm_a else None,
    }
    draws = [r for r in rows if r["artifact"].endswith(("mpc_target_baseline.json",
                                                        "mpc_target_baseline_draw2.json"))]
    agreement = {
        "reach_equal": draws[0]["reach"] == draws[1]["reach"],
        "upright_equal": draws[0]["upright_arrivals"] == draws[1]["upright_arrivals"],
        "band_time_equal": draws[0]["pct_steps_in_band"] == draws[1]["pct_steps_in_band"],
        "share_standing_equal": draws[0]["share_closed_standing_all"]
        == draws[1]["share_closed_standing_all"],
    }
    data = {
        "protocol": ("every row is the same 20 seeded episodes (seed 11..30, reset_mode=mixed, "
                     "task_phase=target) measured by bench_approach_mechanism.trace_episode; the MPC "
                     "cells come from bench_mpc.py and the learned comparators from the committed "
                     "curves, so arrival and gait are read off one instrument on both sides"),
        "mpc_rows": rows, "learned": learned, "baseline_two_draws_agree": agreement,
        "reading": ("the planner with more search beats every learned arm on arrival and still loses on "
                    "the gait measure: it closes a fifth to a third of its ground inside the standing "
                    "band where the learned arms close half to three quarters, and no cell of any kind "
                    "holds the band for two continuous seconds"),
    }
    return data, out



SHAPING_TRAINING_LOG = "chain31_evidence.log"
DONE_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) DONE (.+?) exit=(\d+) seconds=(\d+)$")


def value_shaping_pair(out="benchmarks/value_shaping_pair.json"):
    """The planner's value as a shaping term inside SAC, against the same code with the flag off.

    Every number here is read from a committed file: the wall clocks come from the launcher log, which
    for a SAC run is the only place they exist; the scores come from the eval artifacts; the gait
    measures come from the per-step trace rows the arms were added to. Nothing is typed.

    The block exists to answer two questions at once. One is the experiment - does the shaping help?
    The other is whether the comparison is allowed: the control was re-run by the revision that carries
    `--value-potential`, and its own published twin (same seed, same budget, same world, older code) is
    quoted next to it so the reader can see which halves of a SAC run survive a window.
    """
    clocks = {}
    with open(os.path.join(ROOT, SHAPING_TRAINING_LOG), encoding="utf-8", errors="replace") as handle:
        for line in handle:
            m = DONE_RE.match(line.strip())
            if m:
                clocks[m.group(2).strip()] = {"finished": m.group(1), "exit_code": int(m.group(3)),
                                              "seconds": int(m.group(4))}
    labels = {
        "control_s7": "SAC 5M control seed 7 (unshaped)",
        "shaped_s7": "SAC 5M shaped alpha 0.19 seed 7",
        "shaped_s8": "SAC 5M shaped alpha 0.19 seed 8",
    }
    missing = [label for label in labels.values() if label not in clocks]
    if missing:
        raise FileNotFoundError(f"{SHAPING_TRAINING_LOG} has no completed span for {missing}")

    mech = json.load(open(os.path.join(ROOT, "benchmarks", "approach_mechanism.json"),
                           encoding="utf-8"))
    traces = {(r["artifact"], r["model"]): r for r in mech["rows"]}
    reach = json.load(open(os.path.join(ROOT, "benchmarks", "reach_upright_arrival.json"),
                            encoding="utf-8"))
    upright = {(r["artifact"], r["model"]): r for r in reach["rows"]}
    coverage = json.load(open(os.path.join(ROOT, "benchmarks", "planner_potential_coverage.json"),
                              encoding="utf-8"))
    potential = json.load(open(os.path.join(ROOT, "benchmarks", "planner_potential.json"),
                               encoding="utf-8"))

    cells = {}
    for artifact, model, budget, seed, arm in (
            ("benchmarks/vshape_control_sac_n32_r4_5m.json", "control_s7_5m", 5000000, 7, "control"),
            ("benchmarks/vshape_control_sac_n32_r4_5m.json", "control_s7_3m", 3000000, 7, "control"),
            ("benchmarks/vshape_sac_n32_r4_5m.json", "vshape_s7_5m", 5000000, 7, "shaped"),
            ("benchmarks/vshape_sac_n32_r4_5m.json", "vshape_s7_3m", 3000000, 7, "shaped"),
            ("benchmarks/vshape_sac_n32_r4_5m_seed8.json", "vshape_s8_5m", 5000000, 8, "shaped")):
        ev = json.load(open(os.path.join(ROOT, artifact), encoding="utf-8"))["models"][model]
        trace, arr = traces[(artifact, model)], upright[(artifact, model)]
        cells[model] = {
            "arm": arm, "budget": budget, "seed": seed, "artifact": artifact,
            "checkpoint": ev["checkpoint"], "mean_return": ev["mean"],
            "reached_distance_only_pct": ev["reached_target_pct"],
            "reached_upright_pct": ev["reached_target_upright_pct"],
            "reached_upright_count": arr["reached_upright_count"],
            "falls_per_episode": ev["falls_per_episode"],
            "value_shaping": ev["value_shaping"],
            "wall_clock_seconds": clocks[labels[arm + "_s%d" % seed]]["seconds"] if budget == 5000000
            else None,
            "pct_steps_in_band": trace["mean_pct_steps_in_band"],
            "mean_longest_band_run_s": trace["mean_longest_band_run_s"],
            "reach_episodes": trace["reach_episodes"]["n"],
            "share_closed_standing_in_reach": trace["reach_episodes"]["mean_share_while_standing"],
            "heading_velocity_in_band_reach": trace["reach_episodes"]["mean_heading_velocity_in_band_mps"],
        }

    paired = {}
    for seed in (7, 8):
        for budget in (3000000, 5000000):
            c = [k for k, v in cells.items() if v["arm"] == "control" and v["seed"] == seed
                 and v["budget"] == budget]
            s = [k for k, v in cells.items() if v["arm"] == "shaped" and v["seed"] == seed
                 and v["budget"] == budget]
            if not c or not s:
                continue
            ctrl, shp = cells[c[0]], cells[s[0]]
            paired[f"seed{seed}_{budget // 1000000}m"] = {
                "control": c[0], "shaped": s[0],
                "upright_arrivals": [ctrl["reached_upright_count"], shp["reached_upright_count"]],
                "reached_distance_only_pct": [ctrl["reached_distance_only_pct"],
                                              shp["reached_distance_only_pct"]],
                "pct_steps_in_band": [ctrl["pct_steps_in_band"], shp["pct_steps_in_band"]],
                "share_closed_standing": [ctrl["share_closed_standing_in_reach"],
                                          shp["share_closed_standing_in_reach"]],
                "heading_velocity_in_band": [ctrl["heading_velocity_in_band_reach"],
                                             shp["heading_velocity_in_band_reach"]],
                "mean_return": [ctrl["mean_return"], shp["mean_return"]],
            }

    published = json.load(open(os.path.join(ROOT, "benchmarks", "utd_sac_n32_r4_5m.json"),
                               encoding="utf-8"))["models"]["s5000000"]
    fresh = cells["control_s7_5m"]
    parity = {
        "question": ("does the trainer with the shaping flag available still produce the run the "
                     "published table records, with the flag off?"),
        "published": {"artifact": "benchmarks/utd_sac_n32_r4_5m.json", "model": "s5000000",
                      "mean_return": published["mean"],
                      "reached_distance_only_pct": published["reached_target_pct"],
                      "device": "cuda", "physics_preset": "v9"},
        "rerun_this_revision": {"artifact": fresh["artifact"], "model": "control_s7_5m",
                                "mean_return": fresh["mean_return"],
                                "reached_distance_only_pct": fresh["reached_distance_only_pct"],
                                "reached_upright_count": fresh["reached_upright_count"],
                                "wall_clock_seconds": fresh["wall_clock_seconds"]},
        "loose_arrival_column_equal": published["reached_target_pct"] == fresh["reached_distance_only_pct"],
        "mean_return_ratio": round(fresh["mean_return"] / published["mean"], 3),
        "reading": ("the arrival column reproduced; the return level did not. So the shaped arm is "
                    "compared against this revision's own control, and the published mean is context "
                    "rather than the second half of a ratio"),
    }

    rule = coverage["cells"]["control"]
    return {
        "protocol": ("three SAC runs at 5M, num_envs 32, utd-ratio 4, task_phase target, reset_mode "
                     "mixed, target_forward_velocity 1.2, world v9, trained back to back in one window "
                     "(chain31); scored by eval_phase1.py --num-episodes 20 --seed 11 on the published "
                     "reward - the shaping never enters the environment an arm is graded in - and "
                     "traced step by step by bench_approach_mechanism.py"),
        "training": {key: clocks[value] | {"label": value}
                     for key, value in labels.items()},
        "potential": {"artifact": "benchmarks/planner_potential.json",
                      "r2_holdout": potential["r2_holdout"],
                      "r2_ridge_same_split": potential["r2_ridge_same_split"],
                      "r2_ridge_raw_inputs": potential["r2_ridge_raw_inputs"],
                      "selected_epoch": potential["selected_epoch"],
                      "holdout_seeds": potential["holdout_seeds"]},
        "weight_rule": {"artifact": "benchmarks/planner_potential_coverage.json",
                        "reward_fraction": coverage["reward_fraction"],
                        "mean_abs_task_reward_per_step": rule["mean_abs_task_reward_per_step"],
                        "mean_abs_shaping_at_weight_1": rule["mean_abs_shaping_at_weight_1"],
                        "shaping_to_reward_ratio_at_weight_1":
                            rule["shaping_to_reward_ratio_at_weight_1"],
                        "pct_states_clamped": rule["pct_states_clamped"],
                        "weight_by_rule": rule["weight_by_rule"],
                        "weight_used": fresh_shape_weight(cells)},
        "cells": cells,
        "paired": paired,
        "parity_control_vs_published": parity,
        "shaping_pool": mech["pooled_v9_shaping_all_episodes"],
    }, out


def fresh_shape_weight(cells):
    """The alpha every shaped cell was trained with, read off the checkpoints' own record."""
    weights = {c["value_shaping"]["weight"] for c in cells.values()
               if c["arm"] == "shaped" and c["value_shaping"]}
    if len(weights) != 1:
        raise ValueError(f"the shaped cells were not trained with one weight: {sorted(weights)}")
    return weights.pop()


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
    # The 40M from-scratch target run vs the curriculum run, paired per seeded episode, both
    # scored in one session on one device because the published curve never recorded its device.
    target_from_scratch_paired,
    # The same SAC screen trained in the published world and in the cheap preset: what a 2.7x
    # cheaper env step is worth inside a real loop, and what a 1M screen cannot see.
    physics_preset_screens,
    # The pair re-run at 5M, the smallest budget where the target is actually reached in either
    # world - the only arm that can say whether the cheap world tracks the published one.
    physics_preset_screens_5m,
    # The same two worlds drawn a second time at a second seed, which is what separates the ordering
    # of the worlds from the luck of one run.
    physics_preset_second_draw,
    # A sampling MPC on the same episodes as the learned arms: what simulator access buys on this
    # task, measured twice because the constraint solver is not bit-deterministic.
    mpc_target_rows,
    # The planner's value as an in-loop shaping signal for SAC, against the same code with the flag off,
    # with the parity of the unshaped path and the wall clocks read out of the launcher log.
    value_shaping_pair,
    # PPO against SAC at the same config: what each costs per environment step, and what that buys
    # in behaviour. The two answers point in opposite directions, so they live in one artifact.
    trainer_pair_ppo_sac,
    # The preset on the update-dominated trainer: is the gain predictable from the loop split?
    physics_preset_dreamer_pair,
    # All three physics worlds trained end to end at 1M, so the step ratio is not quoted alone.
    physics_presets_sac_triple,
    # The same optimisation dose bought at 32 envs instead of 8: what the update is really worth.
    utd_dose_pair,
    # That pair drawn a second time at a second seed, which is what makes the reach claim a claim.
    utd_dose_draws,
    # PPO at ten times the budget it was dismissed at, against the clock of every SAC arm.
    ppo_10m_against_sac,
    # That PPO run drawn a second time: the behaviour replicates, the wall clock does not.
    ppo_10m_draws,
    # The third physics world at the budget where reaching is observable, drawn twice.
    euler_screen_5m_draws,
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
