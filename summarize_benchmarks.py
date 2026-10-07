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
    # PPO against SAC at the same config: what each costs per environment step, and what that buys
    # in behaviour. The two answers point in opposite directions, so they live in one artifact.
    trainer_pair_ppo_sac,
    # The preset on the update-dominated trainer: is the gain predictable from the loop split?
    physics_preset_dreamer_pair,
    # All three physics worlds trained end to end at 1M, so the step ratio is not quoted alone.
    physics_presets_sac_triple,
    # The same optimisation dose bought at 32 envs instead of 8: what the update is really worth.
    utd_dose_pair,
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
