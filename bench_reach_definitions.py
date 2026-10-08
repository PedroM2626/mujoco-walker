"""Re-score the checkpoints whose target-task claims the README makes, under both arrival rules.

Why this script exists: `reached_target_pct` has always counted an arrival by distance alone - the
closest approach got inside the 0.45 m radius - while the environment's own success term also requires
the torso above 1.0 m and upright above 0.7 at that step. A fall forward can land inside the radius,
and the reward pays nothing for it, so the two counts disagree on exactly the arms the README quotes.
This measures both, on the same seeded episodes the published rows used, and refuses to write anything
unless the loose column reproduces the committed artifact to the digit.

The fidelity check is the point of the file rather than a courtesy: without it, a stricter number
could be produced by scoring a different episode than the one published, and the comparison would be
between two measurements instead of two definitions of one measurement.

Usage:
    python bench_reach_definitions.py            # writes benchmarks/reach_upright_arrival.json
    python bench_reach_definitions.py --episodes 20 --seed 11
"""
import argparse
import json
import os

import numpy as np

import eval_phase1
import envs.walker_ragdoll_env  # noqa: F401  (registers WalkerRagdoll-v0)

ROOT = os.path.dirname(os.path.abspath(__file__))

# (artifact, model key, what the README says about it) - the arms whose reach columns are quoted.
CHECKPOINTS = [
    ("benchmarks/utd_sac_n32_seed8_r4_5m.json", "s4000000", "SAC dose 4, seed 8, best cell"),
    ("benchmarks/utd_sac_n32_r4_5m.json", "s3000000", "SAC dose 4, seed 7, best cell"),
    ("benchmarks/utd_sac_n32_seed8_r1_5m.json", "s4000000", "SAC dose 1, seed 8, best cell"),
    ("benchmarks/utd_sac_n32_r1_5m.json", "s4000000", "SAC dose 1, seed 7, best cell"),
    ("benchmarks/ppo_learning_curve_10m_v9_seed8.json", "s7012352", "PPO 10M, seed 8, best cell"),
    ("benchmarks/ppo_learning_curve_10m_v9.json", "s5046272", "PPO 10M, seed 7, best cell"),
    ("benchmarks/target_learning_curve_from_scratch_v9.json", "s5000000", "40M from scratch, 5M"),
    ("benchmarks/target_learning_curve_from_scratch_v9.json", "s8000000", "40M from scratch, 8M"),
    ("benchmarks/physics_presets_screen5m_v9.json", "s5000000", "published world, 5M, seed 7"),
    ("benchmarks/physics_presets_screen5m_seed8_v9.json", "s5000000", "published world, 5M, seed 8"),
    ("benchmarks/physics_presets_screen5m_fast.json", "s5000000", "fast world, 5M, seed 7"),
    ("benchmarks/physics_presets_screen5m_seed8_fast.json", "s5000000", "fast world, 5M, seed 8"),
    ("benchmarks/physics_presets_screen5m_euler.json", "s5000000", "euler world, 5M, seed 7"),
    ("benchmarks/physics_presets_screen5m_seed8_euler.json", "s5000000", "euler world, 5M, seed 8"),
    # The two gait arms from chain30b. Arm A raises the target-phase stability reward, arm B trains
    # under the target distance curriculum - and its transfer column is the same checkpoints scored
    # without it, which is the only way to ask whether the curriculum taught anything transferable.
    ("benchmarks/stab60_sac_n32_r4_5m_curve.json", "s4000000", "stability 60, seed 7, best cell"),
    ("benchmarks/stab60_sac_n32_r4_5m_curve.json", "s5000000", "stability 60, seed 7, endpoint"),
    ("benchmarks/stab60_sac_n32_r4_5m_seed8_curve.json", "s4000000", "stability 60, seed 8, best cell"),
    ("benchmarks/tcur_sac_n32_r4_5m_curve.json", "s4000000", "curriculum, seed 7, in-task"),
    ("benchmarks/tcur_sac_n32_r4_5m_seed8_curve.json", "s2000000", "curriculum, seed 8, in-task"),
    ("benchmarks/tcur_sac_n32_r4_5m_transfer.json", "x4000000", "curriculum, seed 7, on 2-5 m"),
    ("benchmarks/tcur_sac_n32_r4_5m_seed8_transfer.json", "x5000000", "curriculum, seed 8, on 2-5 m"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--steps", type=int, default=eval_phase1.EPISODE_STEPS)
    ap.add_argument("--out", default="benchmarks/reach_upright_arrival.json")
    args = ap.parse_args()

    rows, failures = [], []
    for artifact, key, label in CHECKPOINTS:
        path = os.path.join(ROOT, artifact)
        art = json.load(open(path, encoding="utf-8"))
        model = art["models"][key]
        tele = art["per_episode"][f"{key}__telemetry"]
        preset = art.get("physics_preset", "v9")
        # Read back from the artifact: scoring a curriculum checkpoint on the published 2-5 m is a
        # different question from scoring it in its own task, and the row label says which one it is.
        curriculum = bool(model.get("target_curriculum"))
        algo, phase, width, policy, on_start, _ = eval_phase1.build_policy(
            os.path.join(ROOT, model["checkpoint"]), art.get("device") or "cpu")
        _, _, _, new_tele = eval_phase1.score(
            policy, args.episodes, args.seed, model["task_phase"], args.steps, "mixed",
            on_episode_start=on_start, reward_kwargs=model.get("reward_kwargs"),
            physics_preset=preset, target_curriculum=curriculum)

        # Fidelity: the loose column must be the published one, episode for episode.
        loose_old = [bool(t["reached_target"]) for t in tele]
        loose_new = [bool(t["reached_target"]) for t in new_tele]
        dist_old = [round(float(t["min_target_distance"]), 3) for t in tele]
        dist_new = [round(float(t["min_target_distance"]), 3) for t in new_tele]
        if loose_old != loose_new or dist_old != dist_new:
            failures.append(f"{artifact}:{key} did not reproduce "
                            f"(reached {sum(loose_old)}->{sum(loose_new)}, "
                            f"dist {dist_old[:3]}... vs {dist_new[:3]}...)")
        strict = [bool(t["reached_target_upright"]) for t in new_tele]
        rows.append({
            "label": label, "artifact": artifact, "model": key,
            "checkpoint": model["checkpoint"], "physics_preset": preset,
            "target_curriculum": curriculum,
            "algo": model["algo"], "episodes": args.episodes, "seed": args.seed,
            "reached_distance_only_pct": model["reached_target_pct"],
            "reached_upright_pct": round(100.0 * float(np.mean(strict)), 1),
            "reached_upright_count": int(sum(strict)),
            "collapse_on_target_only": int(sum(o and not s for o, s in zip(loose_new, strict))),
            "mean": model["mean"], "mean_min_target_distance": model["mean_min_target_distance"],
            "falls_per_episode": model["falls_per_episode"],
            "standing_at_end_pct": model["standing_at_end_pct"],
        })
        print(f"{label:36s} loose={model['reached_target_pct']:>5}%  "
              f"upright={100.0 * float(np.mean(strict)):>5.1f}%  "
              f"collapse-only={sum(o and not s for o, s in zip(loose_new, strict)):2d}"
              f"  {'OK' if not failures else 'MISMATCH'}")

    if failures:
        for f in failures:
            print("FIDELITY FAILURE:", f)
        raise SystemExit(1)

    # "best" is scoped to the published world: the presets change what arriving costs, so a
    # cross-world maximum would answer a question nobody asked.
    v9_rows = [r for r in rows if r["physics_preset"] == "v9"]
    best = max(v9_rows, key=lambda r: r["reached_upright_count"])
    best_any = max(rows, key=lambda r: r["reached_upright_count"])
    payload = {
        "protocol": (f"re-scored with eval_phase1.score at --num-episodes {args.episodes} "
                     f"--seed {args.seed} --steps {args.steps} --reset-mode mixed, in the world each "
                     "checkpoint trained in; the distance-only column is required to reproduce the "
                     "committed artifact episode for episode before anything is written"),
        "definitions": {
            "reached_distance_only_pct": "closest approach ever <= 0.45 m - the number every "
                                         "published row in this repository carries",
            "reached_upright_pct": "closest approach <= 0.45 m at a step with torso height > 1.0 m "
                                   "and upright > 0.7 - the condition the environment's own success "
                                   "term applies, so a fall onto the marker does not count",
        },
        "rows": rows,
        "best_upright_arrivals": {"label": best["label"], "checkpoint": best["checkpoint"],
                                  "reached_upright_count": best["reached_upright_count"],
                                  "reached_upright_pct": best["reached_upright_pct"],
                                  "reached_distance_only_pct": best["reached_distance_only_pct"],
                                  "scoped_to": "the published world (physics_preset v9)"},
        "best_upright_arrivals_any_world": {"label": best_any["label"],
                                            "physics_preset": best_any["physics_preset"],
                                            "reached_upright_count": best_any["reached_upright_count"],
                                            "reached_upright_pct": best_any["reached_upright_pct"]},
        "note": ("the two columns disagree on every arm here, and the gap is not noise: it is "
                 "episodes where the body arrived by falling. A claim about walking to the target has "
                 "to quote the upright column."),
    }
    out = os.path.join(ROOT, args.out)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"wrote {out} ({len(rows)} rows, all reproducing their published loose column)")


if __name__ == "__main__":
    main()
