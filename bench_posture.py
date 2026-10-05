"""Does the robot actually stand up? The episode return cannot say, so this measures the torso.

Every Phase-1 policy scores 0% "standing at end" and 0.1-0.4 falls per episode, and those two
numbers together are misleading rather than reassuring. `is_healthy` in this environment is
`1.0 < torso z < 2.0`, and termination only latches once an episode has *been* healthy
(`_latch_health`), which exists so that a fallen start is the task and not a failure. The
consequence is that a policy which never raises its torso above 1.0 m "never falls" for a thousand
steps while it stays on the floor - and under v9, where posture is a bonus graded from the floor
up, it is paid for that. So a low fall count is evidence about the health latch, not about standing.

This script reports the height itself, per step: how many episodes ever reach the band, how high the
torso gets, what fraction of the steps sit inside it, and what the same episodes earn.

    python bench_posture.py                                   # default roster, 10 episodes each
    python bench_posture.py --episodes 50 --seed 11
    python bench_posture.py --compare-devices --only-named --model dreamer=checkpoints/...pt
    python bench_posture.py --model name=path.pt --model zero=none   # none|random are references

Writes benchmarks/phase1_posture_probe.json.

Two protocol choices are deliberate and both are recorded in the artifact. The device defaults to
`auto`, which is what `eval_phase1.py --device auto` resolves to here (cuda): at 50 episodes this
script then reproduces the published mean return to the cent, which is the cross-check that makes
the posture columns quotable next to the score. And the torch CPU thread count is pinned, because
with torch's default threading the same checkpoint gave 0.634 m and 0.616 m on two runs of the
identical command.
"""

import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
import envs.walker_ragdoll_env  # noqa: F402  (registers WalkerRagdoll-v0)
import gymnasium as gym  # noqa: E402

STEPS = 1000
BAND = (1.0, 2.0)          # the env's own healthy_z_range
UPRIGHT = 0.8              # the env's own is_upright threshold
THREADS = 1                # see the docstring: default threading was not repeatable run to run

DEFAULT_MODELS = [
    ("dreamer_v3_1m", "checkpoints/dreamer_v3_1m/dreamer_actor_1000000.pt"),
    ("dreamer_v9_269k", "checkpoints/dreamer_v2_1m_v9/dreamer_actor_269404.pt"),
    ("ars_v9_1m", "checkpoints/ars_v2_1m_v9/ars_ckpt_1000917.pt"),
    # Both SAC checkpoints, because the first version of this roster had one entry named `sac_40m`
    # pointing at the 9M file - so the row the README published as "SAC at 40M steps" measured the
    # 9M policy. Every row now records the path it was scored from and a gate compares that path's
    # step count against the label, which is the check that was missing.
    ("sac_40m", "checkpoints/walker_target_v1/sac_ckpt_40000000.pt"),
    ("sac_9m", "checkpoints/walker_target_v1/sac_ckpt_9000000.pt"),
    ("commanding_zero", "none"),
    ("uniform_random", "random"),
]


def resolve_device(requested):
    """`auto` means what `eval_phase1.py --device auto` means on this box: cuda when present."""
    if requested != "auto":
        return requested
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_policy(spec, device, action_dim):
    """(policy(obs)->action, reset hook, reward info) for one roster entry.

    The reward is read off the checkpoint rather than assumed, exactly as `eval_phase1.py` does:
    scoring an agent under a reward it never optimised is how this repository once concluded that
    its walkers "rank by posture".

    The two passive references have no checkpoint to read, so they get the reward every trained row
    gets. Scoring them under the environment defaults instead - which is what the first published
    table did - puts two different reward functions in one column, and the defaults pay
    `standing_reward=50/step` that training sets to 0, so the references that the trained policies
    are ranked against would have been paid for lying still.
    """
    if spec == "none":
        return (lambda obs: np.zeros(action_dim), lambda: None, _reference_reward_info())
    if spec == "random":
        rs = np.random.RandomState(0)
        return (lambda obs: rs.uniform(-1.0, 1.0, size=action_dim), lambda: None,
                _reference_reward_info())
    import torch
    from eval_phase1 import build_policy
    algo, phase, width, policy, reset, rinfo = build_policy(spec, torch.device(device))
    return policy, reset, rinfo


def _reference_reward_info():
    from envs.reward_shaping import TRAINING_REWARD_KWARGS
    return {"reward_kwargs": dict(TRAINING_REWARD_KWARGS),
            "reward_source": "no checkpoint - TRAINING_REWARD_KWARGS, so the return column is one "
                             "reward function across the table"}


def probe(policy, reset, reward_kwargs, episodes, seed, steps, reset_mode, task_phase,
          threads, device):
    """One seeded pass over the band statistics.

    The seeding mirrors `eval_phase1.py`: the RSSM posterior is reparametrised, so an unseeded
    scorer gives a different number for the same checkpoint on every call - the defect that made the
    inert-reference table unreproducible until the policy RNG was seeded per model.
    """
    import torch
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = gym.make("WalkerRagdoll-v0", reset_mode=reset_mode, task_phase=task_phase,
                   **reward_kwargs)
    rows = []
    try:
        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            reset()
            zs, ups, latched, total = [], [], False, 0.0
            n_falls, was_healthy = 0, env.unwrapped.is_healthy
            for _ in range(steps):
                obs, reward, terminated, truncated, info = env.step(
                    np.asarray(policy(obs), dtype=np.float64).reshape(-1))
                total += float(reward)
                zs.append(float(env.unwrapped.data.qpos[2]))
                ups.append(float(env.unwrapped.upright_factor))
                healthy = env.unwrapped.is_healthy
                latched = latched or healthy
                if was_healthy and not healthy:
                    n_falls += 1
                was_healthy = healthy
                if terminated or truncated:
                    break
            zs, ups = np.asarray(zs), np.asarray(ups)
            in_band = (zs >= BAND[0]) & (zs <= BAND[1])
            standing = in_band & (ups > UPRIGHT)
            rows.append({"steps": int(zs.size), "return": round(total, 2),
                         "max_z": round(float(zs.max()), 3),
                         "final_z": round(float(zs[-1]), 3),
                         "pct_steps_in_band": round(100.0 * in_band.mean(), 2),
                         "pct_steps_upright_in_band": round(100.0 * standing.mean(), 2),
                         "mean_upright": round(float(ups.mean()), 3),
                         "ever_healthy": bool(latched), "falls": int(n_falls)})
    finally:
        env.close()
    mean = lambda key: round(float(np.mean([r[key] for r in rows])), 3 if "z" in key else 2)
    return {
        "episodes": len(rows), "device": device,
        "episodes_ever_in_band": sum(1 for r in rows if r["ever_healthy"]),
        "mean_max_z": mean("max_z"),
        "best_max_z_in_one_episode": round(float(max(r["max_z"] for r in rows)), 3),
        "lowest_peak_max_z": round(float(min(r["max_z"] for r in rows)), 3),
        "mean_pct_steps_in_band": mean("pct_steps_in_band"),
        "mean_pct_steps_upright_in_band": mean("pct_steps_upright_in_band"),
        "mean_final_z": mean("final_z"),
        "mean_falls_per_episode": mean("falls"),
        "mean_return": mean("return"),
        "per_episode": rows,
    }


def run_entry(name, path, args, action_dim, device):
    if path not in ("none", "random") and not os.path.exists(path):
        return {"error": f"{path} not in this checkout"}
    policy, reset, rinfo = make_policy(path, device, action_dim)
    rkw = {} if args.reward_weights == "env-default" else dict(rinfo["reward_kwargs"])
    res = probe(policy, reset, rkw, args.episodes, args.seed, args.steps, args.reset_mode,
                args.task_phase, args.threads, device)
    res["reward_source"] = ("forced --reward-weights=env-default" if rkw == {}
                            else rinfo["reward_source"])
    res["reward_kwargs_applied"] = rkw
    # Recorded so a row can never again be labelled with a step count its file does not have.
    res["checkpoint"] = path
    return res


HEADERS = (f"{'model':18}{'ever in band':>14}{'mean max z':>12}{'% steps in band':>17}"
           f"{'% upright':>11}{'final z':>9}{'falls/ep':>9}{'mean return':>13}")


def report(name, res):
    print(f"{name:18}{res['episodes_ever_in_band']:>8}/{res['episodes']:<5}"
          f"{res['mean_max_z']:>12.3f}{res['mean_pct_steps_in_band']:>17.2f}"
          f"{res['mean_pct_steps_upright_in_band']:>11.2f}{res['mean_final_z']:>9.3f}"
          f"{res['mean_falls_per_episode']:>9.2f}{res['mean_return']:>13.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", default=[], metavar="NAME=PATH",
                   help="add or override a roster entry; PATH may be none (zero action) or random")
    p.add_argument("--only-named", action="store_true",
                   help="score only the --model entries given, instead of the roster plus them")
    p.add_argument("--compare-devices", action="store_true",
                   help="run the named entries once per device and print the pair - the posture "
                        "verdict survives the device, the return does not")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--task-phase", default="target")
    p.add_argument("--reset-mode", default="mixed")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="device the policies run on; auto matches eval_phase1.py, and it matters - "
                        "see --compare-devices and benchmarks/phase1_eval_device_sensitivity.json "
                        "for the spread this box produces")
    p.add_argument("--threads", type=int, default=THREADS,
                   help="torch CPU thread count pinned for the run; 0 leaves torch's default")
    p.add_argument("--reward-weights", default="auto", choices=["auto", "env-default"],
                   help="read the reward off each checkpoint (auto, as eval_phase1.py does) or "
                        "score everything with the environment defaults")
    p.add_argument("--out", default=os.path.join("benchmarks", "phase1_posture_probe.json"))
    args = p.parse_args()

    roster, named = dict(DEFAULT_MODELS), []
    for item in args.model:
        name, _, path = item.partition("=")
        roster[name] = path
        named.append(name)
    if args.only_named:
        roster = {n: roster[n] for n in named}

    ref = gym.make("WalkerRagdoll-v0", reset_mode=args.reset_mode, task_phase=args.task_phase)
    action_dim = int(np.prod(ref.action_space.shape))
    ref.close()

    out = {"protocol": f"envs/walker_ragdoll_env.py at ENV_VERSION "
                       f"{envs.walker_ragdoll_env.ENV_VERSION}, task_phase={args.task_phase}, "
                       f"reset_mode={args.reset_mode}, {args.episodes} seeded episodes (seed "
                       f"{args.seed}), {args.steps} steps per episode, standing band = torso z in "
                       f"{BAND} as the env defines it, upright threshold {UPRIGHT} as the env's "
                       f"is_healthy uses it",
           "why": "a low falls_per_episode is not evidence of standing: termination only latches "
                  "after an episode has been healthy, so a policy that never reaches the band "
                  "never registers a fall",
           "action_dim": action_dim, "torch_threads": args.threads}

    if args.compare_devices:
        out["devices"] = {}
        for dev in ("cuda", "cpu"):
            out["devices"][dev] = {n: run_entry(n, roster[n], args, action_dim, dev) for n in named}
        print(f"{'model':18}{'cuda return':>13}{'cpu return':>13}{'spread %':>10}"
              f"{'cuda band%':>12}{'cpu band%':>11}{'band eps cuda/cpu':>19}")
        for name in named:
            c = out["devices"]["cuda"][name]
            p2 = out["devices"]["cpu"][name]
            if "error" in c or "error" in p2:
                print(f"{name:18}  skipped: {c.get('error') or p2.get('error')}")
                continue
            mid = (c["mean_return"] + p2["mean_return"]) / 2.0
            print(f"{name:18}{c['mean_return']:>13.2f}{p2['mean_return']:>13.2f}"
                  f"{100.0 * abs(c['mean_return'] - p2['mean_return']) / mid:>10.1f}"
                  f"{c['mean_pct_steps_in_band']:>12.2f}{p2['mean_pct_steps_in_band']:>11.2f}"
                  f"{str(c['episodes_ever_in_band']) + '/' + str(p2['episodes_ever_in_band']):>19}")
    else:
        out["device"] = resolve_device(args.device)
        out["models"] = {}
        print(HEADERS)
        for name, path in roster.items():
            res = run_entry(name, path, args, action_dim, out["device"])
            if "error" in res:
                print(f"{name:18}  skipped: {res['error']}")
                continue
            out["models"][name] = res
            report(name, res)

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
