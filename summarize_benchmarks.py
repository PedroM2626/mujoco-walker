"""Build machine-readable summaries of the re-measured Phase 3 and Phase 4 benchmarks.

Reads the run logs, so the published numbers cannot drift from what was executed.
Re-run after a benchmark to refresh benchmarks/*.json.
"""
import json
import os
import re

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
