"""A sampling-MPC baseline for the target task, scored by the same instrument as the learned rows.

Why this exists: every target-phase number in the README is a *learned* policy, and the question the
residual-learning idea would have to be measured against - can a planner reach the marker on this
ragdoll at all, and does it walk when it does - has no committed answer. The MJPC binary in
`mujoco_mpc_walker/` cannot provide one on this machine: its `main.cc` parses only `--task` and then
enters `StartApp`, the GUI event loop, so it has no headless episode-scoring mode (and Windows App
Control blocks that particular executable, though not the other built binaries). So the baseline here is
a cross-entropy planner written against the same environment, the same reward, the same 20 seeded
episodes and the same per-step trace as `bench_approach_mechanism.py`.

Three things the rollout loop must not corrupt, each guarded below:
  * the scored episode's step counter - rollouts run on `env.unwrapped`, never through `TimeLimit`;
  * the target itself - `step()` resamples `_target_xy` and bumps `_curriculum_level` on a reach, so a
    rollout that reaches would move the goal out from under the episode being scored;
  * the simulator state - restored from a `mjSTATE_FULLPHYSICS` snapshot before every rollout, which is
    exact, although the constraint solver is not bit-deterministic (two identical rollouts diverge
    around the third step), so the planner searches an approximate model rather than an oracle.

Usage:
    python bench_mpc.py                                    # 20 episodes, CEM 128x30
    python bench_mpc.py --episodes 2 --samples 32          # smoke test
"""
import argparse
import json
import os
import time

import mujoco
import numpy as np

import gymnasium as gym

import envs.walker_ragdoll_env  # noqa: F401  (registers WalkerRagdoll-v0)
from envs.reward_shaping import TRAINING_REWARD_KWARGS
import bench_approach_mechanism as bam

ROOT = os.path.dirname(os.path.abspath(__file__))
FULL = mujoco.mjtState.mjSTATE_FULLPHYSICS


class CrossEntropyPlanner:
    """Receding-horizon CEM over action sequences, scoring them with the environment's own reward.

    The cost is the shaped return, not a hand-written one: giving the planner a different objective than
    the learned policies optimise would make the comparison a comparison of tasks, and the whole point is
    to ask what a controller with simulator access achieves on *this* task.
    """

    def __init__(self, horizon=30, samples=128, iterations=4, elite=16, replan=5,
                 init_std=0.6, min_std=0.05, seed=0, action="argmin"):
        self.horizon, self.samples, self.iterations = horizon, samples, iterations
        self.elite, self.replan, self.init_std, self.min_std = elite, replan, init_std, min_std
        # "argmin" returns the winner of the search; "elite-mean" returns the mean of the elite set's
        # first action, which is the standard policy-extraction trick for MPC and the thing a
        # behaviour-cloning student can actually regress: the argmin flips between near-tied
        # sequences, and measured on 28k teacher steps consecutive argmin actions differ by MSE 0.757
        # against the action's own energy of 0.381, i.e. the recorded policy is close to white noise.
        self.action_mode = action
        self.rng = np.random.default_rng(seed)
        self.env = None
        self.pending = []
        self.step_index = 0
        self.plan_ms = []
        self.rollout_steps = 0
        self.last_value = 0.0

    # -- binding to the episode being scored -----------------------------------------------
    def bind(self, env):
        self.env = env
        self.pending = []
        self.step_index = 0
        self.last_value = 0.0
        # plan_ms and rollout_steps accumulate across episodes on purpose: they are cost counters,
        # and resetting them here made the artifact report one episode's totals as the run's.
        if not hasattr(self, "plan_ms"):
            self.plan_ms, self.rollout_steps = [], 0

    def _snapshot(self):
        unw = self.env.unwrapped
        buf = np.zeros(mujoco.mj_stateSize(unw.model, FULL), dtype=np.float64)
        mujoco.mj_getState(unw.model, unw.data, buf, FULL)
        # The task state that step() mutates on a reach; rollouts must not move the goal.
        return buf, int(unw._curriculum_level), np.asarray(unw._target_xy).copy(), \
            bool(unw._ever_healthy)

    def _restore(self, snap):
        buf, level, target, ever = snap
        unw = self.env.unwrapped
        mujoco.mj_setState(unw.model, unw.data, buf, FULL)
        mujoco.mj_forward(unw.model, unw.data)
        unw._curriculum_level = level
        unw._target_xy = np.asarray(target, dtype=np.float64)
        unw._ever_healthy = ever

    # -- the planner ------------------------------------------------------------------------
    def _evaluate(self, plan):
        """Return of one candidate action sequence from the current state, then restore."""
        snap = self._snapshot()
        total = 0.0
        for action in plan:
            _, reward, terminated, _, _ = self.env.unwrapped.step(action)
            total += float(reward)
            self.rollout_steps += 1
            if terminated:
                break
        self._restore(snap)
        return total

    def plan(self):
        t0 = time.perf_counter()
        unw = self.env.unwrapped
        level_before = unw._curriculum_level
        target_before = np.asarray(unw._target_xy, dtype=np.float64).copy()
        dim = int(np.prod(unw.action_space.shape))
        # Warm start: the previous best sequence shifted by the steps already committed.
        prior = np.zeros((self.horizon, dim))
        carried = self.pending[self.replan:] if self.pending else []
        for i, a in enumerate(carried[:self.horizon]):
            prior[i] = a
        mean, std = prior.copy(), np.full((self.horizon, dim), self.init_std)
        best_seq, best_ret, elite_first = None, -np.inf, None
        for _ in range(self.iterations):
            draws = np.clip(mean + std * self.rng.standard_normal(
                (self.samples, self.horizon, dim)), -1.0, 1.0)
            returns = np.empty(self.samples)
            for i in range(self.samples):
                returns[i] = self._evaluate(draws[i])
            keep = np.argsort(-returns)[:self.elite]
            mean = draws[keep].mean(axis=0)
            std = np.maximum(draws[keep].std(axis=0), self.min_std)
            elite_first = draws[keep][:, 0, :].mean(axis=0)
            if returns[keep[0]] > best_ret:
                best_ret, best_seq = float(returns[keep[0]]), draws[keep[0]]
        if self.action_mode == "elite-mean":
            # The committed first action is the elite average; the plan that continues the rollouts
            # stays the winner's, so the search itself is unchanged and only the executed action is
            # smoothed.
            best_seq = best_seq.copy()
            best_seq[0] = elite_first
        if unw._curriculum_level != level_before or not np.allclose(unw._target_xy, target_before):
            raise RuntimeError(
                "a rollout moved the episode's target: the restore is incomplete, so the plan is "
                "being scored against a goal that the search itself relocated")
        self.plan_ms.append((time.perf_counter() - t0) * 1e3)
        return best_seq, best_ret

    # -- the policy interface the trace expects ---------------------------------------------
    def __call__(self, obs):
        if self.step_index % self.replan == 0 or not self.pending:
            seq, value = self.plan()
            self.pending = [a.copy() for a in seq]
            # The plan's lookahead return, kept until the next replan. It is the teacher's own opinion
            # about the state it is standing in, and unlike the executed action it is a scalar the
            # student could regress - which is what makes it worth recording.
            self.last_value = float(value)
        action = self.pending.pop(0)
        self.step_index += 1
        return action


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--iterations", type=int, default=4)
    ap.add_argument("--elite", type=int, default=16)
    ap.add_argument("--replan", type=int, default=5)
    ap.add_argument("--action", choices=["argmin", "elite-mean"], default="argmin",
                    help="which action the plan executes: the search winner, or the mean of the elite "
                         "set's first action. The second is the smooth target a behaviour-cloning "
                         "student can regress - the winner flips between near-tied sequences.")
    ap.add_argument("--device", default="cpu", help="the planner is one env; the GPU is for the "
                                                   "learned arms, and mixing them is what made two "
                                                   "earlier rate claims false")
    ap.add_argument("--out", default="benchmarks/mpc_target_baseline.json")
    args = ap.parse_args()

    planner = CrossEntropyPlanner(horizon=args.horizon, samples=args.samples,
                                  iterations=args.iterations, elite=args.elite,
                                  replan=args.replan, seed=args.seed, action=args.action)
    env = gym.make("WalkerRagdoll-v0", reset_mode="mixed", task_phase="target",
                   **TRAINING_REWARD_KWARGS)
    radius = float(env.unwrapped._target_radius)
    traces = []
    t0 = time.perf_counter()
    for ep in range(args.episodes):
        trace = bam.trace_episode(env, planner, lambda: None, args.seed, ep, args.steps, radius,
                                  on_env=planner.bind)
        traces.append(trace)
        print(f"ep{ep:2d} seed{trace['seed']:3d} closest={trace['min_target_distance']:6.3f} "
              f"upright_arr={str(trace['reached_target_upright']):5s} "
              f"band={trace['pct_steps_in_band']:5.1f}% longest_run={trace['longest_band_run_s']:4.2f}s "
              f"falls={trace['n_falls']:2d} plans={len(planner.plan_ms):4d}")
    wall = time.perf_counter() - t0
    env.close()

    reach = [t for t in traces if t["reached_target"]]
    upright = [t for t in traces if t["reached_target_upright"]]
    ms = np.asarray(planner.plan_ms, dtype=float) if planner.plan_ms else np.zeros(1)
    # One plan costs samples*iterations*horizon env steps; the rate is stated per simulated step so it
    # can be compared with the 1,560 steps/s a single unwrapped env collects at.
    steps_per_plan = args.samples * args.iterations * args.horizon
    payload = {
        "protocol": (f"cross-entropy sampling MPC over WalkerRagdoll-v0 task_phase=target, "
                     f"horizon={args.horizon} steps ({args.horizon * 0.01:.2f} s), "
                     f"samples={args.samples}, cem_iterations={args.iterations}, "
                     f"elite={args.elite}, replan every {args.replan} steps "
                     f"({args.replan * 0.01:.2f} s), cost = the environment's own shaped reward "
                     f"(envs.reward_shaping.TRAINING_REWARD_KWARGS), one env on {args.device}; "
                     f"scored with {args.episodes} episodes, seed {args.seed}, reset_mode=mixed - "
                     "the same protocol and the same per-step trace as the learned rows"),
        "measured_with": "bench_approach_mechanism.trace_episode / .pooled",
        "config": {"action": args.action, "horizon": args.horizon, "samples": args.samples,
                   "iterations": args.iterations, "elite": args.elite, "replan": args.replan,
                   "episodes": args.episodes, "seed": args.seed, "device": args.device},
        "cost": {"wall_clock_s": round(wall, 1),
                 "plan_ms_mean": round(float(ms.mean()), 1),
                 "plan_ms_p95": round(float(np.percentile(ms, 95)), 1),
                 "sim_steps_per_s_planning": round(1e3 / (ms.mean() / steps_per_plan), 1),
                 "rollout_steps_total": int(planner.rollout_steps),
                 "rollout_steps_per_episode": round(float(planner.rollout_steps) / args.episodes, 1),
                 "plans_total": int(len(ms)),
                 "plans_per_episode": round(float(len(ms)) / args.episodes, 1),
                 "planning_s_per_episode": round(float(ms.sum()) / 1e3 / args.episodes, 1)},
        "all_episodes": bam.pooled(traces),
        "reach_episodes": bam.pooled(reach) if reach else None,
        "upright_arrival_episodes": bam.pooled(upright) if upright else None,
        "per_episode": traces,
        "caveats": [
            "the constraint solver is not bit-deterministic: two rollouts from an identical restored "
            "state diverge within a few steps, so the planner searches an approximate model",
            "the planner is given the reward the learned policies are scored on, including the "
            "standing gate that zeroes every locomotion term below the band",
            "one environment, no vectorisation: the planning budget below is what a single CPU core "
            "affords, not what MJPC or a GPU batch would cost",
        ],
    }
    out = os.path.join(ROOT, args.out)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(json.dumps({"reach": payload["reach_episodes"], "upright": payload["upright_arrival_episodes"],
                      "cost": payload["cost"]}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
