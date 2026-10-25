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
    python scripts/bench_mpc.py                                    # 20 episodes, CEM 128x30
    python scripts/bench_mpc.py --episodes 2 --samples 32          # smoke test
"""
import argparse
import json
import os
import time

import mujoco
import numpy as np

import gymnasium as gym

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import envs.walker_ragdoll_env  # noqa: F401  (registers WalkerRagdoll-v0)
from envs.reward_shaping import TRAINING_REWARD_KWARGS
import bench_approach_mechanism as bam

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULL = mujoco.mjtState.mjSTATE_FULLPHYSICS


class CrossEntropyPlanner:
    """Receding-horizon CEM over action sequences, scoring them with the environment's own reward.

    The cost is the shaped return, not a hand-written one: giving the planner a different objective than
    the learned policies optimise would make the comparison a comparison of tasks, and the whole point is
    to ask what a controller with simulator access achieves on *this* task.
    """

    def __init__(self, horizon=30, samples=128, iterations=4, elite=16, replan=5,
                 init_std=0.6, min_std=0.05, seed=0, action="argmin",
                 objective="published", hold=1, cont_decay=1.0, slew_weight=0.0,
                 stability_weight=None, strict_band_gate=False,
                 settled_fall_termination=False, overspeed_weight=0.0,
                 target_vel=0.8, band_bonus=0.0):
        self.horizon, self.samples, self.iterations = horizon, samples, iterations
        self.elite, self.replan, self.init_std, self.min_std = elite, replan, init_std, min_std
        # "argmin" returns the winner of the search; "elite-mean" returns the mean of the elite set's
        # first action, which is the standard policy-extraction trick for MPC and the thing a
        # behaviour-cloning student can actually regress: the argmin flips between near-tied
        # sequences, and measured on 28k teacher steps consecutive argmin actions differ by MSE 0.757
        # against the action's own energy of 0.381, i.e. the recorded policy is close to white noise.
        self.action_mode = action
        self.objective = objective
        self.hold = max(1, int(hold))
        self.cont_decay = float(cont_decay)
        if objective == "gait":
            self.slew_weight = 0.35 if slew_weight == 0.0 else float(slew_weight)
            self.stability_weight = 60.0 if stability_weight is None else float(stability_weight)
            self.strict_band_gate = True
            self.settled_fall_termination = bool(settled_fall_termination)
            self.overspeed_weight = 80.0 if overspeed_weight == 0.0 else float(overspeed_weight)
            self.target_vel = float(target_vel)
            self.band_bonus = 20.0 if band_bonus == 0.0 else float(band_bonus)
        else:
            self.slew_weight = float(slew_weight)
            self.stability_weight = stability_weight
            self.strict_band_gate = bool(strict_band_gate)
            self.settled_fall_termination = bool(settled_fall_termination)
            self.overspeed_weight = float(overspeed_weight)
            self.target_vel = float(target_vel)
            self.band_bonus = float(band_bonus)
        self.rng = np.random.default_rng(seed)
        self.env = None
        self.pending = []
        self.step_index = 0
        self.plan_ms = []
        self.rollout_steps = 0
        self.last_value = 0.0
        self.last_action = None
        # Off by default: recording is a collector-side convenience, and bench_mpc's own runs must keep
        # the exact cost they publish. When on, every plan leaves its candidate first-actions and the
        # return the search assigned them - a sample of Q(s, a) that the CEM was already computing and
        # throwing away. The action it executes is the argmin of these samples and is not imitable;
        # whether their value is, is the question bench_planner_q.py answers.
        self.record_candidates = False
        self.last_candidates = None
        self.last_obs = None
        self.just_replanned = False

    # -- binding to the episode being scored -----------------------------------------------
    def bind(self, env):
        self.env = env
        self.pending = []
        self.step_index = 0
        self.last_value = 0.0
        self.last_action = np.zeros(int(np.prod(env.unwrapped.action_space.shape)), dtype=np.float64)
        # plan_ms and rollout_steps accumulate across episodes on purpose: they are cost counters,
        # and resetting them here made the artifact report one episode's totals as the run's.
        if not hasattr(self, "plan_ms"):
            self.plan_ms, self.rollout_steps = [], 0

    def _snapshot(self):
        unw = self.env.unwrapped
        buf = np.zeros(mujoco.mj_stateSize(unw.model, FULL), dtype=np.float64)
        mujoco.mj_getState(unw.model, unw.data, buf, FULL)
        # The task state that step() mutates on a reach; rollouts must not move the goal.
        return (buf, int(unw._curriculum_level), np.asarray(unw._target_xy).copy(),
                bool(unw._ever_healthy), bool(unw._terminate_when_unhealthy),
                float(unw._stability_reward_weight))

    def _restore(self, snap):
        buf, level, target, ever, term_flag, stab_w = snap
        unw = self.env.unwrapped
        mujoco.mj_setState(unw.model, unw.data, buf, FULL)
        mujoco.mj_forward(unw.model, unw.data)
        unw._curriculum_level = level
        unw._target_xy = np.asarray(target, dtype=np.float64)
        unw._ever_healthy = ever
        unw._terminate_when_unhealthy = term_flag
        unw._stability_reward_weight = stab_w

    def _expand_knots(self, knots):
        if self.hold <= 1:
            return knots[:, :self.horizon, :]
        return np.repeat(knots, self.hold, axis=1)[:, :self.horizon, :]

    # -- the planner ------------------------------------------------------------------------
    def _evaluate(self, plan, settled_standing=False):
        """Return of one candidate action sequence from the current state, then restore."""
        snap = self._snapshot()
        unw = self.env.unwrapped
        if self.settled_fall_termination:
            unw._terminate_when_unhealthy = bool(settled_standing)
        if self.stability_weight is not None:
            unw._stability_reward_weight = float(self.stability_weight)
        total = 0.0
        prev_a = self.last_action
        use_gait_terms = (self.strict_band_gate or self.band_bonus > 0.0
                          or self.overspeed_weight > 0.0 or self.slew_weight > 0.0)
        for action in plan:
            _, reward, terminated, _, info = unw.step(action)
            r = float(reward)
            if use_gait_terms:
                z = float(unw.data.qpos[2])
                up = float(unw.upright_factor)
                in_band = (z > bam.BAND_Z) and (up > bam.BAND_UPRIGHT)
                if self.strict_band_gate and not in_band:
                    # Zero locomotion terms below the standing band (0.65 < z <= 1.0): the environment's
                    # standing_gate starts at z = 0.65 m, which pays up to 58.3% of locomotion reward
                    # while the torso is already below the 1.0 m standing band.
                    r -= float(info.get("reward_target_progress", 0.0))
                    r -= float(info.get("reward_target_direction", 0.0))
                    r -= float(info.get("reward_forward_velocity", 0.0))
                if in_band:
                    if self.band_bonus > 0.0:
                        height_margin = min(max((z - bam.BAND_Z) / 0.25, 0.0), 1.0)
                        r += self.band_bonus * height_margin * (up ** 2)
                    if self.overspeed_weight > 0.0:
                        v_toward = float(info.get("velocity_toward_target", 0.0))
                        if v_toward > self.target_vel:
                            r -= self.overspeed_weight * ((v_toward - self.target_vel) ** 2)
                if self.slew_weight > 0.0 and prev_a is not None:
                    diff = action - prev_a
                    r -= self.slew_weight * float(np.dot(diff, diff))
                prev_a = action
            total += r
            self.rollout_steps += 1
            if terminated:
                break
        self._restore(snap)
        return total

    def plan(self):
        t0 = time.perf_counter()
        self.last_candidates = [] if self.record_candidates else None
        unw = self.env.unwrapped
        level_before = unw._curriculum_level
        target_before = np.asarray(unw._target_xy, dtype=np.float64).copy()
        z0 = float(unw.data.qpos[2])
        up0 = float(unw.upright_factor)
        settled_standing = (z0 > 1.12) and (up0 > 0.80)
        dim = int(np.prod(unw.action_space.shape))
        if self.hold <= 1 and self.cont_decay == 1.0 and self.slew_weight == 0.0:
            # Exact published path: 100 Hz independent draws, bit-for-bit RNG consumption.
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
                    returns[i] = self._evaluate(draws[i], settled_standing)
                keep = np.argsort(-returns)[:self.elite]
                if self.last_candidates is not None:
                    # First action of each sampled sequence, with the return that sequence earned from this
                    # state. Later iterations are warm-started around the elite, so they are not i.i.d.
                    # samples - the probe groups by plan and never treats them as independent draws.
                    self.last_candidates.append(
                        np.column_stack([draws[:, 0, :], returns]).astype(np.float64))
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
        else:
            n_knots = int(np.ceil(self.horizon / self.hold))
            prior = np.zeros((n_knots, dim))
            carried = self.pending[self.replan:] if self.pending else []
            for k in range(n_knots):
                idx = k * self.hold
                if idx < len(carried):
                    prior[k] = carried[idx]
                elif self.last_action is not None:
                    prior[k] = self.last_action * (0.8 ** (k + 1))
            mean = prior.copy()
            decay = (self.cont_decay ** np.arange(n_knots))[:, None]
            std = np.maximum(self.init_std * decay, self.min_std)
            best_seq, best_ret, elite_first = None, -np.inf, None
            for _ in range(self.iterations):
                knot_draws = np.clip(mean + std * self.rng.standard_normal(
                    (self.samples, n_knots, dim)), -1.0, 1.0)
                draws = self._expand_knots(knot_draws)
                returns = np.empty(self.samples)
                for i in range(self.samples):
                    returns[i] = self._evaluate(draws[i], settled_standing)
                keep = np.argsort(-returns)[:self.elite]
                if self.last_candidates is not None:
                    self.last_candidates.append(
                        np.column_stack([knot_draws[:, 0, :], returns]).astype(np.float64))
                mean = knot_draws[keep].mean(axis=0)
                std = np.maximum(knot_draws[keep].std(axis=0), self.min_std)
                elite_first = knot_draws[keep][:, 0, :].mean(axis=0)
                if returns[keep[0]] > best_ret:
                    best_ret, best_seq = float(returns[keep[0]]), draws[keep[0]].copy()
            if self.action_mode == "elite-mean":
                best_seq = best_seq.copy()
                best_seq[:self.hold] = elite_first
        if unw._curriculum_level != level_before or not np.allclose(unw._target_xy, target_before):
            raise RuntimeError(
                "a rollout moved the episode's target: the restore is incomplete, so the plan is "
                "being scored against a goal that the search itself relocated")
        if self.last_candidates is not None:
            self.last_candidates = np.concatenate(self.last_candidates, axis=0)
        self.plan_ms.append((time.perf_counter() - t0) * 1e3)
        return best_seq, best_ret

    # -- the policy interface the trace expects ---------------------------------------------
    def __call__(self, obs):
        self.just_replanned = False
        if self.record_candidates:
            # The state these candidates were scored from, kept beside them so the probe can group by
            # state rather than fit across a shuffled pool. The planner restores the env after every
            # candidate rollout, so this observation is the one the search started from.
            self.last_obs = np.asarray(obs, dtype=np.float32).copy()
        if self.step_index % self.replan == 0 or not self.pending:
            seq, value = self.plan()
            self.pending = [a.copy() for a in seq]
            self.just_replanned = True
            # The plan's lookahead return, kept until the next replan. It is the teacher's own opinion
            # about the state it is standing in, and unlike the executed action it is a scalar the
            # student could regress - which is what makes it worth recording.
            self.last_value = float(value)
        action = self.pending.pop(0)
        self.last_action = np.asarray(action, dtype=np.float64).copy()
        self.step_index += 1
        return action


def _bench_episode_worker(task):
    ep, cfg = task
    planner = CrossEntropyPlanner(
        horizon=cfg["horizon"], samples=cfg["samples"], iterations=cfg["iterations"],
        elite=cfg["elite"], replan=cfg["replan"], seed=cfg["seed"] + ep,
        action=cfg["action"], objective=cfg["objective"], hold=cfg["hold"],
        init_std=cfg["init_std"], min_std=cfg["min_std"], cont_decay=cfg["cont_decay"],
        slew_weight=cfg["slew_weight"],
        settled_fall_termination=cfg["settled_fall_termination"],
        overspeed_weight=cfg["overspeed_weight"], target_vel=cfg["target_vel"],
        band_bonus=cfg["band_bonus"],
    )
    env = gym.make("WalkerRagdoll-v0", reset_mode="mixed", task_phase="target",
                   **TRAINING_REWARD_KWARGS)
    try:
        radius = float(env.unwrapped._target_radius)
        trace = bam.trace_episode(env, planner, lambda: None, cfg["seed"], ep, cfg["steps"], radius,
                                  on_env=planner.bind)
        return trace, list(planner.plan_ms), int(planner.rollout_steps)
    finally:
        env.close()


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
    ap.add_argument("--objective", choices=["published", "gait"], default="published",
                    help="'published' scores rollouts with the unmodified environment reward; 'gait' "
                         "zeroes locomotion below the z > 1.0 m standing band, raises stability weight "
                         "to 60, and bounds forward velocity so the search optimises walking rather "
                         "than a ballistic lunge.")
    ap.add_argument("--hold", type=int, default=1,
                    help="steps each sampled control knot is held for (default 1 = 100 Hz white noise).")
    ap.add_argument("--init-std", type=float, default=0.6)
    ap.add_argument("--min-std", type=float, default=0.05)
    ap.add_argument("--cont-decay", type=float, default=1.0,
                    help="per-knot std decay across the horizon (default 1.0 = uniform std).")
    ap.add_argument("--slew-weight", type=float, default=0.0,
                    help="quadratic penalty on consecutive action differences ||a_t - a_{t-1}||^2.")
    ap.add_argument("--overspeed-weight", type=float, default=0.0,
                    help="in-band penalty on velocity toward target above --target-vel.")
    ap.add_argument("--target-vel", type=float, default=0.8,
                    help="reference forward velocity for the gait objective.")
    ap.add_argument("--band-bonus", type=float, default=0.0,
                    help="in-band height-margin bonus weight.")
    ap.add_argument("--settled-fall-termination", action="store_true",
                    help="enable terminate_when_unhealthy inside candidate rollouts that start settled "
                         "in the standing band (z > 1.12 m, upright > 0.80).")
    ap.add_argument("--workers", type=int, default=1,
                    help="episode worker processes (1 = exact sequential single-RNG path).")
    ap.add_argument("--device", default="cpu", help="the planner is one env; the GPU is for the "
                                                   "learned arms, and mixing them is what made two "
                                                   "earlier rate claims false")
    ap.add_argument("--out", default="benchmarks/mpc_target_baseline.json")
    args = ap.parse_args()

    traces = []
    t0 = time.perf_counter()
    if args.workers <= 1:
        planner = CrossEntropyPlanner(horizon=args.horizon, samples=args.samples,
                                      iterations=args.iterations, elite=args.elite,
                                      replan=args.replan, seed=args.seed, action=args.action,
                                      objective=args.objective, hold=args.hold,
                                      init_std=args.init_std, min_std=args.min_std,
                                      cont_decay=args.cont_decay, slew_weight=args.slew_weight,
                                      settled_fall_termination=args.settled_fall_termination,
                                      overspeed_weight=args.overspeed_weight,
                                      target_vel=args.target_vel, band_bonus=args.band_bonus)
        env = gym.make("WalkerRagdoll-v0", reset_mode="mixed", task_phase="target",
                       **TRAINING_REWARD_KWARGS)
        radius = float(env.unwrapped._target_radius)
        for ep in range(args.episodes):
            trace = bam.trace_episode(env, planner, lambda: None, args.seed, ep, args.steps, radius,
                                      on_env=planner.bind)
            traces.append(trace)
            print(f"ep{ep:2d} seed{trace['seed']:3d} closest={trace['min_target_distance']:6.3f} "
                  f"upright_arr={str(trace['reached_target_upright']):5s} "
                  f"band={trace['pct_steps_in_band']:5.1f}% longest_run={trace['longest_band_run_s']:4.2f}s "
                  f"falls={trace['n_falls']:2d} plans={len(planner.plan_ms):4d}")
        env.close()
        plan_ms_all = planner.plan_ms
        rollout_steps_total = int(planner.rollout_steps)
    else:
        from concurrent.futures import ProcessPoolExecutor
        cfg = vars(args).copy()
        plan_ms_all = []
        rollout_steps_total = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for ep, (trace, p_ms, r_steps) in enumerate(
                    pool.map(_bench_episode_worker, [(ep, cfg) for ep in range(args.episodes)])):
                traces.append(trace)
                plan_ms_all.extend(p_ms)
                rollout_steps_total += r_steps
                print(f"ep{ep:2d} seed{trace['seed']:3d} closest={trace['min_target_distance']:6.3f} "
                      f"upright_arr={str(trace['reached_target_upright']):5s} "
                      f"band={trace['pct_steps_in_band']:5.1f}% longest_run={trace['longest_band_run_s']:4.2f}s "
                      f"falls={trace['n_falls']:2d} plans={len(plan_ms_all):4d}", flush=True)
    wall = time.perf_counter() - t0

    reach = [t for t in traces if t["reached_target"]]
    upright = [t for t in traces if t["reached_target_upright"]]
    ms = np.asarray(plan_ms_all, dtype=float) if plan_ms_all else np.zeros(1)
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
        "config": {"action": args.action, "objective": args.objective, "hold": args.hold,
                   "cont_decay": args.cont_decay, "slew_weight": args.slew_weight,
                   "settled_fall_termination": args.settled_fall_termination,
                   "overspeed_weight": args.overspeed_weight, "target_vel": args.target_vel,
                   "band_bonus": args.band_bonus, "horizon": args.horizon, "samples": args.samples,
                   "iterations": args.iterations, "elite": args.elite, "replan": args.replan,
                   "episodes": args.episodes, "seed": args.seed, "device": args.device},
        "cost": {"wall_clock_s": round(wall, 1),
                 "plan_ms_mean": round(float(ms.mean()), 1),
                 "plan_ms_p95": round(float(np.percentile(ms, 95)), 1),
                 "sim_steps_per_s_planning": round(1e3 / (ms.mean() / steps_per_plan), 1),
                 "rollout_steps_total": int(rollout_steps_total),
                 "rollout_steps_per_episode": round(float(rollout_steps_total) / args.episodes, 1),
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
