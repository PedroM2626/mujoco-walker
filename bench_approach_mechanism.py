"""Ask whether a "reach" episode is locomotion or a forward topple, per step instead of per episode.

`reached_target_upright_pct` closes one gap - an episode that arrives with the torso on the floor no
longer counts - but it says nothing about how the body got to the marker: an episode can stand up next
to the target after toppling across the room, and it is then counted as an upright arrival while the
forward motion was entirely a fall. The README's claim that the agent "walks" rests on
`mean_x_velocity`, which is an episode mean and is exactly what a repeated forward collapse produces.

So this script keeps the whole per-step trace and splits every metric at the step level by whether the
robot was in the standing band (torso z > 1.0 m and upright > 0.7 - the band the environment's own
success term uses) at that step:

  * `closing_while_standing_m` / `closing_while_down_m` - how much of the distance to the target was
    closed in band and how much outside it. A walker closes it in band.
  * `heading_velocity_in_band_mps` - the component of the root velocity along the direction of the
    target, averaged over the in-band steps only. This is the one that distinguishes "standing and
    moving toward the target" from "standing still while upright terms pay".
  * `pct_steps_in_band` and `min_dist_after_first_fall` - when the approach happened relative to the
    falls.

The fidelity gate is the same as `bench_reach_definitions.py`'s: the run has to reproduce the committed
artifact's per-episode closest approach, both arrival flags, step count and x-velocity exactly, or it
writes nothing. Otherwise the mechanism would be measured on different episodes than the ones the
published rows describe.

Usage:
    python bench_approach_mechanism.py                      # all 14 quoted checkpoints
    python bench_approach_mechanism.py --only utd_sac_n32_seed8_r4_5m.json:s4000000
"""
import argparse
import json
import os

import numpy as np

import eval_phase1
import envs.walker_ragdoll_env  # noqa: F401  (registers WalkerRagdoll-v0)

ROOT = os.path.dirname(os.path.abspath(__file__))

# Same rows as bench_reach_definitions.py: every checkpoint whose reach column the README quotes.
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
    # The two gait arms (chain30b). Arm A raises the stability reward; arm B trains under the target
    # distance curriculum, and its transfer column is the same checkpoints scored without it.
    ("benchmarks/stab60_sac_n32_r4_5m_curve.json", "s4000000", "stability 60, seed 7, best cell"),
    ("benchmarks/stab60_sac_n32_r4_5m_curve.json", "s5000000", "stability 60, seed 7, endpoint"),
    ("benchmarks/stab60_sac_n32_r4_5m_seed8_curve.json", "s4000000", "stability 60, seed 8, best cell"),
    ("benchmarks/tcur_sac_n32_r4_5m_curve.json", "s4000000", "curriculum, seed 7, in-task"),
    ("benchmarks/tcur_sac_n32_r4_5m_seed8_curve.json", "s2000000", "curriculum, seed 8, in-task"),
    ("benchmarks/tcur_sac_n32_r4_5m_transfer.json", "x4000000", "curriculum, seed 7, on 2-5 m"),
    ("benchmarks/tcur_sac_n32_r4_5m_seed8_transfer.json", "x5000000", "curriculum, seed 8, on 2-5 m"),
    # The offline-critic students: trained on the planner's transitions, scored like everything else.
    ("benchmarks/critic_mpc_students.json", "critic_q", "offline critic, Q-ascent actor"),
    ("benchmarks/critic_mpc_students.json", "critic_q_bc", "offline critic, TD3+BC actor"),
    # The potential-shaping pair, trained by the revision that carries --value-potential: the control is
    # the same configuration with the flag off, so the parity of the unshaped path is measured here and
    # not assumed.
    ("benchmarks/vshape_control_sac_n32_r4_5m.json", "control_s7_5m", "control 5M, seed 7, this revision"),
    ("benchmarks/vshape_control_sac_n32_r4_5m.json", "control_s7_3m", "control 3M, seed 7, this revision"),
    ("benchmarks/vshape_sac_n32_r4_5m.json", "vshape_s7_5m", "shaping 0.19, seed 7, 5M"),
    ("benchmarks/vshape_sac_n32_r4_5m.json", "vshape_s7_3m", "shaping 0.19, seed 7, 3M"),
    ("benchmarks/vshape_sac_n32_r4_5m_seed8.json", "vshape_s8_5m", "shaping 0.19, seed 8, 5M"),
]

BAND_Z, BAND_UPRIGHT = 1.0, 0.7

# The four episodes that were rendered into video for the README's "watch it" claim, so the sentences
# that describe those clips are gated on the same traces as the aggregate numbers.
RENDERED_CLIPS = [
    ("benchmarks/utd_sac_n32_seed8_r4_5m.json", "s4000000", 0, "near-miss"),
    ("benchmarks/utd_sac_n32_seed8_r4_5m.json", "s4000000", 3, "reached"),
    ("benchmarks/utd_sac_n32_seed8_r4_5m.json", "s4000000", 14, "reached"),
    ("benchmarks/utd_sac_n32_seed8_r4_5m.json", "s4000000", 18, "dawdling"),
]


def trace_episode(env, policy, on_start, seed, ep, steps, radius, on_env=None):
    """Replay one scored episode and split its progress between the standing band and outside it.

    `on_env` is called with the environment right after the reset, before any step: a controller that
    plans against the simulator's state rather than the observation (the sampling MPC in
    `bench_mpc.py`) needs to bind to the episode's own env, and the alternative - a second env - would
    measure a different trajectory than the one being scored.
    """
    obs, info = env.reset(seed=seed + ep)
    on_start()
    if on_env is not None:
        on_env(env)
    unw = env.unwrapped
    target = np.asarray(unw._target_xy, dtype=np.float64)
    d0 = float(info["target_distance"])
    z0 = float(info["z_position"])
    up0 = float(info["upright"])

    dist_prev = d0
    closing_band = closing_down = 0.0
    head_band, head_down, n_band, n_down = [], [], 0, 0
    v_x_band, v_x_down = [], []
    min_dist, min_step, min_z, min_upright = d0, 0, z0, up0
    # The environment's own arrival, tracked the way eval_phase1.score() tracks it: the closest
    # approach among the steps that were inside the band, which need not be the episode's overall
    # closest approach - a fall can get nearer to the marker than anything standing did.
    min_dist_band, band_closing_at_band_arrival = float("inf"), 0.0
    total_band_closing_at_min = 0.0
    total_down_closing_at_min = 0.0
    n_steps, n_falls, first_fall_step = 0, 0, None
    vel_sum = 0.0
    was_healthy = unw.is_healthy
    # How the time in band is distributed. Ten seconds of standing is a walker; eight runs of 0.3 s
    # separated by collapses is a get-up-and-fall-over process, and the two can close the same
    # distance and produce the same episode-mean x-velocity.
    band_run = band_run_max = band_entries = 0
    was_band = False

    for _ in range(steps):
        action = np.asarray(policy(obs), dtype=np.float64).reshape(-1)
        obs, reward, terminated, truncated, info = env.step(action)
        n_steps += 1
        vel_sum += float(info.get("x_velocity", 0.0))
        dist = float(info["target_distance"])
        z = float(unw.data.qpos[2])
        up = float(unw.upright_factor)
        in_band = z > BAND_Z and up > BAND_UPRIGHT
        closing = dist_prev - dist
        dist_prev = dist
        if in_band:
            n_band += 1
            closing_band += closing
            band_run += 1
            band_run_max = max(band_run_max, band_run)
            if not was_band:
                band_entries += 1
        else:
            n_down += 1
            closing_down += closing
            band_run = 0
        was_band = in_band

        # Velocity component along the (fixed, since the target is resampled only at reset) heading.
        root = np.asarray(unw.data.qpos[0:2], dtype=np.float64)
        delta = target - root
        norm = float(np.hypot(delta[0], delta[1])) or 1.0
        heading = (delta / norm) @ np.asarray(unw.data.qvel[0:2], dtype=np.float64)
        (head_band if in_band else head_down).append(heading)
        (v_x_band if in_band else v_x_down).append(float(info["x_velocity"]))

        if in_band and dist < min_dist_band:
            min_dist_band, band_closing_at_band_arrival = dist, closing_band

        if dist < min_dist:
            min_dist, min_step, min_z, min_upright = dist, n_steps, z, up
            total_band_closing_at_min = closing_band
            total_down_closing_at_min = closing_down

        healthy = unw.is_healthy
        if was_healthy and not healthy:
            n_falls += 1
            if first_fall_step is None:
                first_fall_step = n_steps
        was_healthy = healthy
        if terminated:
            break

    approach = d0 - min_dist
    band_closing_to_min = total_band_closing_at_min
    band_approach = (d0 - min_dist_band) if np.isfinite(min_dist_band) else None
    final_speed = float(np.hypot(unw.data.qvel[0], unw.data.qvel[1]))
    return {
        "seed": seed + ep,
        "z_at_reset": round(z0, 3),
        "start_in_band": bool(z0 > BAND_Z and up0 > BAND_UPRIGHT),
        "steps": n_steps,
        "d0": round(d0, 3),
        "min_target_distance": round(min_dist, 3),
        "reached_target": bool(min_dist <= radius),
        "final_distance": round(float(dist), 3),
        "final_root_speed_mps": round(final_speed, 4),
        "seconds_between_arrival_and_end": round((n_steps - min_step) * 0.01, 2),
        "min_target_distance_upright": (round(min_dist_band, 3)
                                        if np.isfinite(min_dist_band) else None),
        "reached_target_upright": bool(min_dist_band <= radius),
        "min_dist_step": min_step,
        "min_dist_z": round(min_z, 3),
        "min_dist_upright": round(min_upright, 3),
        "n_falls": n_falls,
        "first_fall_step": first_fall_step,
        "approach_distance_m": round(approach, 3),
        "closing_while_standing_m": round(band_closing_to_min, 3),
        "closing_while_down_m": round(approach - band_closing_to_min, 3),
        "share_of_approach_while_standing": (round(band_closing_to_min / approach, 3)
                                             if approach > 0.05 else None),
        # The same split measured up to the in-band arrival instead of up to the closest approach,
        # so an episode that stands next to the marker after toppling onto it cannot be read as a
        # standing approach.
        "band_approach_distance_m": (round(band_approach, 3) if band_approach is not None else None),
        "closing_while_standing_to_band_arrival": (round(band_closing_at_band_arrival, 3)
                                                   if band_approach is not None else None),
        "share_of_band_approach_while_standing": (
            round(band_closing_at_band_arrival / band_approach, 3)
            if band_approach is not None and band_approach > 0.05 else None),
        "pct_steps_in_band": round(100.0 * n_band / max(n_band + n_down, 1), 1),
        "longest_band_run_s": round(band_run_max * 0.01, 2),
        "band_entries": band_entries,
        "heading_velocity_in_band_mps": (round(float(np.mean(head_band)), 4) if head_band else None),
        "heading_velocity_down_mps": (round(float(np.mean(head_down)), 4) if head_down else None),
        "x_velocity_in_band_mps": (round(float(np.mean(v_x_band)), 4) if v_x_band else None),
        "mean_x_velocity": round(vel_sum / max(n_steps, 1), 4),
    }


def pooled(traces):
    runs = [t["longest_band_run_s"] for t in traces]
    closing_s = sum(t["closing_while_standing_m"] for t in traces)
    closing_d = sum(t["closing_while_down_m"] for t in traces)
    return {
        "episodes": len(traces),
        "max_longest_band_run_s": round(max(runs), 2) if runs else None,
        "episodes_with_band_run_ge_2s": sum(1 for x in runs if x >= 2.0),
        "episodes_never_in_band": sum(1 for t in traces if t["pct_steps_in_band"] == 0.0),
        "mean_pct_steps_in_band": _mean([t["pct_steps_in_band"] for t in traces]),
        "closing_while_standing_m": round(closing_s, 2),
        "closing_while_down_m": round(closing_d, 2),
        "share_while_standing": (round(closing_s / (closing_s + closing_d), 3)
                                 if closing_s + closing_d > 0.05 else None),
        "closest_approach_after_first_fall": sum(
            1 for t in traces if t["first_fall_step"] is not None
            and t["min_dist_step"] > t["first_fall_step"]),
    }


def _mean(values):
    vals = [v for v in values if v is not None]
    return round(float(np.mean(vals)), 4) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--steps", type=int, default=eval_phase1.EPISODE_STEPS)
    ap.add_argument("--only", action="append", default=[], metavar="ARTIFACT:MODEL",
                    help="restrict to one row, e.g. benchmarks/x.json:s4000000")
    ap.add_argument("--traces-out", default=None,
                    help="optional JSON with the per-episode traces (not committed by default)")
    ap.add_argument("--out", default="benchmarks/approach_mechanism.json")
    args = ap.parse_args()

    import gymnasium as gym

    wanted = set(args.only)
    rows, failures, all_traces = [], [], {}
    for artifact, key, label in CHECKPOINTS:
        if wanted and f"{artifact}:{key}" not in wanted:
            continue
        path = os.path.join(ROOT, artifact)
        art = json.load(open(path, encoding="utf-8"))
        model = art["models"][key]
        old = art["per_episode"][f"{key}__telemetry"]
        preset = art.get("physics_preset", "v9")
        device = art.get("device") or "cpu"
        # The task the artifact was scored under, read back rather than assumed: a curriculum
        # checkpoint traced without its curriculum is a different MDP, and its transfer column is the
        # same checkpoints traced with it off. The row label says which one this is.
        curriculum = bool(model.get("target_curriculum"))
        algo, phase, width, policy, on_start, _ = eval_phase1.build_policy(
            os.path.join(ROOT, model["checkpoint"]), device)

        env = gym.make("WalkerRagdoll-v0", reset_mode="mixed", task_phase="target",
                       **(model.get("reward_kwargs") or {}),
                       **({} if preset == "v9" else {"physics_preset": preset}),
                       **({"target_curriculum": True} if curriculum else {}))
        radius = float(env.unwrapped._target_radius)
        traces = [trace_episode(env, policy, on_start, args.seed, ep, args.steps, radius)
                  for ep in range(args.episodes)]
        env.close()

        # Fidelity: these are the published episodes, so every number the artifact carries has to
        # come out of this trace loop unchanged.
        for i, (t, o) in enumerate(zip(traces, old)):
            for field in ("steps", "min_target_distance", "reached_target", "mean_x_velocity",
                          "reached_target_upright"):
                if o.get(field) is None or t.get(field) is None:
                    continue
                if abs(float(t[field]) - float(o[field])) > (1e-3 if field != "steps" else 0):
                    failures.append(f"{artifact}:{key} ep{i} {field} {t[field]} != {o[field]}")

        loose = [t for t in traces if t["reached_target"]]
        strict = [t for t in traces if t["reached_target_upright"]]
        rows.append({
            "label": label, "artifact": artifact, "model": key,
            "checkpoint": model["checkpoint"], "physics_preset": preset, "algo": algo,
            # The gait arms are a separate comparison: pooling them with the previously published rows
            # would silently move every figure the existing paragraph quotes. The shaping pair is a
            # third group, for the same reason - and its control belongs with it, not with the published
            # pool, because it is the run that says the unshaped path still does what it did.
            "arm_class": ("gait" if os.path.basename(artifact).startswith(
                ("stab60_", "tcur_", "critic_")) else
                "shaping" if os.path.basename(artifact).startswith("vshape_") else ""),
            "target_curriculum": curriculum,
            # Carried so a shaped arm is recognisable in the trace itself. None also means "saved before
            # the shaping existed", which can only mean unshaped.
            "value_shaping": model.get("value_shaping"),
            "device": device, "episodes": args.episodes, "seed": args.seed,
            "reached_distance_only": len(loose), "reached_upright": len(strict),
            # Which episodes, by index and by the seed the protocol used (seed + ep): this is what a
            # clip has to be selected from, so the choice is reproducible from the committed artifact
            # instead of from a scratch re-run.
            "reach_episode_indices": [i for i, t in enumerate(traces) if t["reached_target"]],
            "upright_arrival_indices": [i for i, t in enumerate(traces)
                                        if t["reached_target_upright"]],
            "mean_start_z": _mean([t["z_at_reset"] for t in traces]),
            "episodes_starting_in_band": sum(1 for t in traces if t["start_in_band"]),
            "mean_pct_steps_in_band": _mean([t["pct_steps_in_band"] for t in traces]),
            "mean_longest_band_run_s": _mean([t["longest_band_run_s"] for t in traces]),
            "mean_band_entries_per_episode": _mean([t["band_entries"] for t in traces]),
            "mean_falls_per_episode": _mean([t["n_falls"] for t in traces]),
            # The headline pair: over the episodes the loose rule calls a reach, how much of the
            # approach was covered standing and how much while the torso was below the band.
            "reach_episodes": {
                "n": len(loose),
                "mean_approach_m": _mean([t["approach_distance_m"] for t in loose]),
                "mean_closing_while_standing_m": _mean([t["closing_while_standing_m"]
                                                        for t in loose]),
                "mean_closing_while_down_m": _mean([t["closing_while_down_m"] for t in loose]),
                "mean_share_while_standing": _mean([t["share_of_approach_while_standing"]
                                                    for t in loose]),
                "closest_approach_after_first_fall": sum(
                    1 for t in loose if t["first_fall_step"] is not None
                    and t["min_dist_step"] > t["first_fall_step"]),
                "mean_heading_velocity_in_band_mps": _mean([t["heading_velocity_in_band_mps"]
                                                            for t in loose]),
                "mean_longest_band_run_s": _mean([t["longest_band_run_s"] for t in loose]),
                "mean_band_entries": _mean([t["band_entries"] for t in loose]),
            },
            "upright_arrival_episodes": {
                "n": len(strict),
                "mean_share_while_standing": _mean([t["share_of_approach_while_standing"]
                                                    for t in strict]),
                "mean_closing_while_standing_m": _mean([t["closing_while_standing_m"]
                                                        for t in strict]),
                "mean_closing_while_down_m": _mean([t["closing_while_down_m"] for t in strict]),
                "mean_heading_velocity_in_band_mps": _mean([t["heading_velocity_in_band_mps"]
                                                            for t in strict]),
                "closest_approach_after_first_fall": sum(
                    1 for t in strict if t["first_fall_step"] is not None
                    and t["min_dist_step"] > t["first_fall_step"]),
                # How the upright arrivals themselves got there: distance closed in band up to the
                # in-band arrival, against the total distance that arrival took.
                "mean_band_approach_m": _mean([t["band_approach_distance_m"] for t in strict]),
                "mean_closing_in_band_to_arrival_m": _mean(
                    [t["closing_while_standing_to_band_arrival"] for t in strict]),
                "mean_share_of_band_approach_while_standing": _mean(
                    [t["share_of_band_approach_while_standing"] for t in strict]),
                "mean_longest_band_run_s": _mean([t["longest_band_run_s"] for t in strict]),
                "mean_band_entries_per_episode": _mean([t["band_entries"] for t in strict]),
            },
            "all_episodes_mean_heading_velocity_in_band_mps": _mean(
                [t["heading_velocity_in_band_mps"] for t in traces]),
        })
        all_traces[f"{artifact}:{key}"] = traces
        r = rows[-1]
        print(f"{label:36s} reach={len(loose):2d}/{args.episodes} "
              f"upright={len(strict):2d} stepsInBand={r['mean_pct_steps_in_band']}% "
              f"approachStanding={r['reach_episodes']['mean_share_while_standing']} "
              f"vHeadingBand={r['reach_episodes']['mean_heading_velocity_in_band_mps']} "
              f"afterFall={r['reach_episodes']['closest_approach_after_first_fall']}"
              f"  {'OK' if not failures else 'MISMATCH'}")

    if failures:
        for f in failures[:20]:
            print("FIDELITY FAILURE:", f)
        raise SystemExit(1)

    v9 = [r for r in rows if r["physics_preset"] == "v9" and not r["arm_class"]]
    v9_gait = [r for r in rows if r["physics_preset"] == "v9" and r["arm_class"] == "gait"]
    v9_shaping = [r for r in rows if r["physics_preset"] == "v9" and r["arm_class"] == "shaping"]
    v9_traces = [t for r in rows if r["physics_preset"] == "v9" and not r["arm_class"]
                 for t in all_traces[f"{r['artifact']}:{r['model']}"]]
    v9_reach = [t for t in v9_traces if t["reached_target"]]
    gait_traces = [t for r in v9_gait for t in all_traces[f"{r['artifact']}:{r['model']}"]]
    gait_reach = [t for t in gait_traces if t["reached_target"]]
    shaping_traces = [t for r in v9_shaping for t in all_traces[f"{r['artifact']}:{r['model']}"]]
    shaping_reach = [t for t in shaping_traces if t["reached_target"]]

    clips = []
    for artifact, key, ep, tag in RENDERED_CLIPS:
        traces = all_traces.get(f"{artifact}:{key}")
        if traces is None or ep >= len(traces):
            continue
        clips.append(dict(traces[ep], rendered_as=tag, artifact=artifact, model=key))

    payload = {
        "protocol": (f"per-step traces of the published seeded episodes (--num-episodes "
                     f"{args.episodes} --seed {args.seed} --steps {args.steps} reset_mode=mixed), "
                     "scored in the world each checkpoint trained in, on the device its artifact "
                     "recorded; nothing is written unless every episode reproduces the committed "
                     "artifact's steps, closest approach, both arrival flags and x-velocity"),
        "definitions": {
            "standing_band": f"torso z > {BAND_Z} m and upright > {BAND_UPRIGHT} at that step - the "
                             "posture condition the environment's own success term applies",
            "closing_while_standing_m": "distance to the target removed at steps inside the band, "
                                        "accumulated up to the episode's closest approach",
            "heading_velocity_in_band_mps": "mean over in-band steps of the root velocity component "
                                            "along the direction of the target - what 'walking "
                                            "toward the target' would have to show up as",
            "longest_band_run_s": "length of the longest contiguous stretch of steps inside the band, "
                                  "at 0.01 s per env step - a walker has to stay in the band for "
                                  "several seconds to cross 2-5 m, so this is the number that says "
                                  "whether the forward motion is a gait or a series of topples",
            "share_while_standing": "pooled over episodes: total metres of distance-to-target closed "
                                    "at in-band steps, divided by the total closed (in band plus "
                                    "outside it)",
        },
        "rendered_clips": clips,
        "rows": rows,
        "pooled_v9_all_episodes": pooled(v9_traces),
        "pooled_v9_reach_episodes": pooled(v9_reach),
        "pooled_v9_gait_rows": [r["label"] for r in v9_gait],
        "pooled_v9_gait_all_episodes": pooled(gait_traces),
        "pooled_v9_gait_reach_episodes": pooled(gait_reach) if gait_reach else None,
        "pooled_v9_shaping_rows": [r["label"] for r in v9_shaping],
        "pooled_v9_shaping_all_episodes": pooled(shaping_traces),
        "pooled_v9_shaping_reach_episodes": pooled(shaping_reach) if shaping_reach else None,
        "pooled_all_rows_reach_episodes": pooled(
            [t for r in rows for t in all_traces[f"{r['artifact']}:{r['model']}"]
             if t["reached_target"]]),
        "summary_v9": {
            "rows": len(v9),
            "reach_episodes": sum(r["reach_episodes"]["n"] for r in v9),
            "upright_arrivals": sum(r["upright_arrival_episodes"]["n"] for r in v9),
            "mean_share_of_approach_while_standing": _mean(
                [r["reach_episodes"]["mean_share_while_standing"] for r in v9
                 if r["reach_episodes"]["n"]]),
            "closest_approach_after_first_fall": sum(
                r["reach_episodes"]["closest_approach_after_first_fall"] for r in v9),
            "mean_pct_steps_in_band": _mean([r["mean_pct_steps_in_band"] for r in v9]),
            "mean_longest_band_run_s": _mean([r["mean_longest_band_run_s"] for r in v9]),
            "mean_heading_velocity_in_band_mps": _mean(
                [r["all_episodes_mean_heading_velocity_in_band_mps"] for r in v9]),
            "upright_arrivals_share_of_band_approach_while_standing": _mean(
                [r["upright_arrival_episodes"]["mean_share_of_band_approach_while_standing"]
                 for r in v9 if r["upright_arrival_episodes"]["n"]]),
        },
    }
    out = os.path.join(ROOT, args.out)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    if args.traces_out:
        with open(os.path.join(ROOT, args.traces_out), "w", encoding="utf-8") as handle:
            json.dump(all_traces, handle, indent=2, ensure_ascii=False)
        print(f"wrote traces to {args.traces_out}")
    print(f"wrote {out} ({len(rows)} rows, all reproducing their published telemetry)")


if __name__ == "__main__":
    main()
