"""Headless evaluation of the Phase-1 online-RL checkpoints (WalkerRagdoll-v0).

Phase 1 had training logs and `--num-episodes` visualisers but no way to *score* a Phase-1
checkpoint: `evaluate_merging.py` knows only the two SAC experts and the merged models, and
`eval_headless.py` belongs to the Phase-2 MPC teacher. So the three Phase-1 trainers could
produce evidence of executing, but never a number comparable to the rest of this repository.

The protocol is deliberately identical to the Phase-3 table in the README - one environment
step per action, `reset_mode="mixed"`, `task_phase="target"`, resets `seed+i`, the
checkpoint's own `obs_rms` applied before the network sees anything - so a row produced here
can sit next to the supervisor/MoE/merged rows without a footnote. Per-episode returns are
written to JSON, which is what makes paired comparison against another checkpoint possible
(see `paired_stats.py` for the same design in Phase 4).

Usage:
    python eval_phase1.py --model redq=checkpoints/redq_v2_1m/redq_actor_1000000.pt \
                          --model dreamer=checkpoints/dreamer_v2_1m/dreamer_actor_1000000.pt \
                          --num-episodes 100 --seed 11
"""

import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym  # noqa: E402
import envs.walker_ragdoll_env  # noqa: E402,F401  (registers WalkerRagdoll-v0)
from envs.reward_shaping import TRAINING_REWARD_KWARGS, reward_kwargs_for  # noqa: E402
from envs.walker_ragdoll_env import ENV_VERSION  # noqa: E402
from envs import normalize_compat  # noqa: F401,E402  pickle shim for obs_rms
from evaluate_merging import EPSILON, CLIP, _policy_input  # noqa: E402
from train_walker import SACAgent  # noqa: E402

ACTION_SPACE = gym.spaces.Box(-1.0, 1.0, shape=(17,))
EPISODE_STEPS = 1000
ALIAS_MARK = "EVAL_PHASE1_ENV_COMMIT"


def reward_info_for(ck):
    """Which reward the checkpoint was trained under, plus the env revision it recorded.

    Scoring an agent with a reward function it never optimised is how "the walkers rank by
    posture" happened (see README): the evaluators used the environment defaults while
    train_walker used its own shaping kwargs. The answer is read off the checkpoint.
    """
    kwargs, source = reward_kwargs_for(ck)
    return {"reward_kwargs": kwargs, "reward_source": source,
            "env_version": ck.get("env_version"), "target_forward_velocity":
            ck.get("target_forward_velocity")}


def _noop_reset():
    pass


def _width_from(state, key="backbone.0.weight"):
    return int(state[key].shape[1])


def _last_linear_index(state, prefix):
    """Highest `net.<i>.weight` index, so an output layer is found by shape and not by luck."""
    idx = [int(k.split(".")[1]) for k in state
           if k.startswith(prefix) and k.endswith(".weight") and k.count(".") == 2]
    if not idx:
        raise SystemExit(f"no {prefix}<i>.weight keys in this checkpoint")
    return max(idx)


def build_policy(path, device):
    """Return (label, policy(obs)->action) for a Phase-1 checkpoint of any algorithm."""
    ck = torch.load(path, map_location=device, weights_only=False)
    algo = str(ck.get("algo", "unknown")).lower()
    phase = ck.get("task_phase")

    if algo == "ars":
        weights = np.asarray(ck["weights"])
        bias = np.asarray(ck["bias"])
        rms = ck.get("obs_rms") or {}

        def ars_policy(obs):
            x = np.asarray(obs, dtype=np.float64)[:weights.shape[1]]
            if rms:
                x = np.clip((x - np.asarray(rms["mean"])) /
                            np.sqrt(np.asarray(rms["var"]) + EPSILON), -CLIP, CLIP)
            # tanh, not clip: train_ars.run_episode acts with np.tanh(w @ obs + b), and its own
            # periodic evaluation (line 231, update_normalizer=False) is the reference this
            # scorer has to reproduce. Clipping keeps the gain at 1 through the linear region
            # where tanh has already compressed it - and a linear policy is only as good as
            # its gain.
            return np.tanh(x @ weights.T + bias)
        return algo, phase, int(weights.shape[1]), ars_policy, _noop_reset, reward_info_for(ck)

    if algo == "dreamer":
        from train_dreamer import DreamerActor, WorldModel

        enc = ck["encoder_state_dict"]
        rssm_sd = ck["rssm_state_dict"]
        actor_sd = ck["actor_state_dict"]
        obs_dim = int(enc["0.weight"].shape[1])
        # nn.GRUCell stores weight_hh as (3*hidden, hidden): reading row 0 gave 768 for a
        # 256-unit GRU, which is what produced a nonsense negative z-dim the first time.
        hidden = int(rssm_sd["gru_cell.weight_hh"].shape[1])
        # net.0 consumes [h, z], so its width minus h is z's; the last layer emits mean+logstd.
        stochastic = int(actor_sd["net.0.weight"].shape[1]) - hidden
        action_dim = int(actor_sd[f"net.{_last_linear_index(actor_sd, 'net.')}.weight"].shape[0]) // 2
        gru_in = int(rssm_sd["gru_cell.weight_ih"].shape[1])
        if gru_in != stochastic + action_dim:
            raise SystemExit(
                f"{path}: inferred widths disagree (RSSM gru input {gru_in} != stochastic "
                f"{stochastic} + action {action_dim}); this checkpoint was written by a "
                "different architecture than train_dreamer.py defines")

        model = WorldModel(obs_dim, action_dim, hidden_dim=hidden,
                           stochastic_dim=stochastic).to(device)
        model.encoder.load_state_dict(enc)
        model.rssm.load_state_dict(rssm_sd)
        actor = DreamerActor(hidden_dim=hidden, stochastic_dim=stochastic,
                             action_dim=action_dim).to(device)
        actor.load_state_dict(actor_sd)
        model.eval()
        actor.eval()
        rms = ck.get("obs_rms")
        # The RSSM state belongs to the policy, so it has to return to zeros at every episode
        # boundary; otherwise an episode is scored carrying the previous episode's belief.
        zero = (torch.zeros(1, hidden, device=device),
                torch.zeros(1, stochastic, device=device),
                torch.zeros(1, action_dim, device=device))
        state = {"h": zero[0], "z": zero[1], "a": zero[2]}

        def reset():
            state["h"], state["z"], state["a"] = (t.clone() for t in zero)

        def dreamer_policy(obs):
            x = np.asarray(obs, dtype=np.float64)[:obs_dim]
            if rms is not None:
                x = np.clip((x - np.asarray(rms.mean)) / np.sqrt(np.asarray(rms.var) + EPSILON),
                            -CLIP, CLIP)
            with torch.no_grad():
                embed = model.encoder(torch.FloatTensor(x).unsqueeze(0).to(device))
                h, _prior, _, _ = model.rssm.transition(state["h"], state["z"], state["a"])
                z, _, _ = model.rssm.posterior(h, embed)
                action = actor.get_action(h, z, sample=False)
            state["h"], state["z"], state["a"] = h, z, action
            return action.squeeze(0).cpu().numpy()
        return algo, phase, obs_dim, dreamer_policy, reset, reward_info_for(ck)

    # sac / redq / td3 / ppo-style actor: either a full state dict or a checkpoint holding one
    sd = ck.get("actor_state_dict", ck)
    agent = SACAgent(_width_from(sd), ACTION_SPACE).to(device)
    agent.load_state_dict(sd)
    agent.eval()
    rms = ck.get("obs_rms")
    width = _width_from(sd)

    def actor_policy(obs):
        tensor = _policy_input(agent, obs, rms, width, device)
        with torch.no_grad():
            return agent.get_action(tensor, deterministic=True)[0].cpu().numpy().reshape(-1)
    label = algo if algo != "unknown" else "sac_actor"
    return label, phase, width, actor_policy, _noop_reset, reward_info_for(ck)


def score(policy, episodes, seed, task_phase, steps=EPISODE_STEPS, reset_mode="mixed",
          on_episode_start=_noop_reset, reward_kwargs=None):
    """Run `episodes` seeded episodes and return (reward, falls, standing, telemetry).

    The telemetry row answers the question the return value cannot: did the robot actually
    get to the target, how close did it come, and how many of the 1000 available env steps
    did it need.
    """
    env = gym.make("WalkerRagdoll-v0", reset_mode=reset_mode, task_phase=task_phase,
                   **(reward_kwargs or {}))
    radius = float(env.unwrapped._target_radius)
    rewards, falls, standing, tele = [], [], [], []
    try:
        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            on_episode_start()
            total, n_falls = 0.0, 0
            n_steps = 0
            dist = float(info.get("target_distance", np.inf))
            min_dist = dist
            vel_sum = 0.0
            was_healthy = env.unwrapped.is_healthy
            for _ in range(steps):
                action = np.asarray(policy(obs), dtype=np.float64).reshape(-1)
                obs, reward, terminated, truncated, info = env.step(action)
                n_steps += 1
                total += float(reward)
                if task_phase == "target":
                    dist = float(info.get("target_distance", np.inf))
                    min_dist = min(min_dist, dist)
                    vel_sum += float(info.get("x_velocity", 0.0))
                is_healthy = env.unwrapped.is_healthy
                if was_healthy and not is_healthy:
                    n_falls += 1
                was_healthy = is_healthy
                if terminated:
                    break
            rewards.append(total)
            falls.append(n_falls)
            standing.append(bool(env.unwrapped.is_healthy
                                 and env.unwrapped.upright_factor > 0.8))
            tele.append({
                "steps": n_steps,
                "min_target_distance": round(min_dist, 3),
                "reached_target": bool(min_dist <= radius),
                "mean_x_velocity": round(vel_sum / max(n_steps, 1), 4),
            } if task_phase == "target" else {
                "steps": n_steps, "min_target_distance": None,
                "reached_target": None, "mean_x_velocity": None,
            })
    finally:
        env.close()
    return rewards, falls, standing, tele


def _relaunch_with_env(commit, argv):
    """Re-run this script with an older env revision aliased as envs.walker_ragdoll_env.

    Some checkpoints record an `env_version` that no longer exists in the repo - the 40M SAC run
    in checkpoints/walker_target_v1 was trained against standup_balance_walk_curriculum_v4, whose
    last revision is 8d37846. Scoring them against today's v8 mixes two MDPs, so the same
    protocol can be run against the revision the checkpoint was built for. The alias has to be in
    sys.modules before the registration import at the top of this file, which means the swap can
    only happen by starting the process again with a prelude.
    """
    root = os.path.dirname(os.path.abspath(__file__))
    out = subprocess.run(["git", "-c", f"safe.directory={root}", "show",
                          f"{commit}:envs/walker_ragdoll_env.py"],
                         cwd=root, capture_output=True, text=True)
    if out.returncode != 0 or "WalkerRagdollEnv" not in out.stdout:
        raise SystemExit(f"rev {commit} não tem envs/walker_ragdoll_env.py: {out.stderr[:200]}")
    # The env resolves its model as dirname(dirname(__file__))/walker_ragdoll.xml, so the copy
    # has to sit one directory deep, exactly where the real envs/ module sits.
    tmp = os.path.join(root, "envs", f"_eval_env_{commit[:8]}.py")
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        handle.write(out.stdout)
    script = (
        "import importlib.util, runpy, sys\n"
        f"spec = importlib.util.spec_from_file_location('envs.walker_ragdoll_env', {tmp!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "sys.modules['envs.walker_ragdoll_env'] = mod\n"
        "spec.loader.exec_module(mod)\n"
        "print('[ENV] aliased revision " + commit + "', mod.ENV_VERSION)\n"
        f"sys.argv = ['eval_phase1.py'] + {argv!r}\n"
        f"runpy.run_path({os.path.join(root, 'eval_phase1.py')!r}, run_name='__main__')\n")
    try:
        os.environ[ALIAS_MARK] = commit
        return subprocess.run([sys.executable, "-u", "-c", script], cwd=root).returncode
    finally:
        os.environ.pop(ALIAS_MARK, None)
        os.remove(tmp)


def main():
    p = argparse.ArgumentParser(description="Avalia checkpoints da Fase 1 no protocolo da Fase 3.")
    p.add_argument("--model", action="append", default=[], metavar="NAME=PATH",
                   help="checkpoint a avaliar; repetivel para uma tabela")
    p.add_argument("--num-episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--task-phase", default=None,
                   help="override; padrao = a fase gravada no proprio checkpoint")
    p.add_argument("--reset-mode", default="mixed")
    p.add_argument("--reward-weights", default="auto", choices=["auto", "training", "env-default"],
                   help="auto = a recompensa gravada no checkpoint (ou a do train_walker para "
                        "checkpoints SAC antigos); training/env-default forcam um dos dois")
    p.add_argument("--steps", type=int, default=EPISODE_STEPS)
    p.add_argument("--env-commit", default=None, metavar="REV",
                   help="avalia contra a versao do ambiente nesse commit (re-executa o script)")
    p.add_argument("--out", default=os.path.join("benchmarks", "phase1_results.json"))
    args = p.parse_args()

    if args.env_commit and not os.environ.get(ALIAS_MARK):
        raise SystemExit(_relaunch_with_env(args.env_commit, sys.argv[1:]))

    if not args.model:
        raise SystemExit("nada a avaliar: passe --model NAME=caminho ( repetivel )")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results, per_episode = {}, {}
    for spec in args.model:
        if "=" not in spec:
            raise SystemExit(f"--model precisa de nome=caminho, recebido {spec!r}")
        name, path = spec.split("=", 1)
        if not os.path.exists(path):
            print(f"[SKIP] {name}: checkpoint ausente {path}")
            continue
        algo, phase, width, policy, reset, rinfo = build_policy(path, device)
        rkw, rsrc = dict(rinfo["reward_kwargs"]), rinfo["reward_source"]
        if args.reward_weights != "auto":
            rkw = {} if args.reward_weights == "env-default" else dict(TRAINING_REWARD_KWARGS)
            rsrc = f"forced --reward-weights={args.reward_weights}"
        if rinfo["env_version"] and rinfo["env_version"] != ENV_VERSION:
            print(f"[NOTE] {name}: checkpoint salvo com env_version={rinfo['env_version']!r}, "
                  f"este repo e {ENV_VERSION!r}; os retornos abaixo sao da env atual. Para o MDP "
                  f"nativo use --env-commit <rev>.")
        task_phase = args.task_phase or phase or "target"
        # The Dreamer actor samples its stochastic state, so scoring it twice gives two numbers
        # (the same checkpoint measured -3425.86, -3637.40 and -3278.44 in three unseeded runs).
        # Seed the policy RNG per model, exactly as openai_walker/evaluate_all.py had to.
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if args.task_phase is None and phase and phase != "target":
            print(f"[NOTE] {name} foi treinado em task_phase={phase}; avaliando em "
                  f"{task_phase} (use --task-phase para mudar)")
        rewards, falls, standing, tele = score(policy, args.num_episodes, args.seed, task_phase,
                                               args.steps, args.reset_mode, on_episode_start=reset,
                                               reward_kwargs=rkw)
        arr = np.asarray(rewards, dtype=float)
        step_match = re.search(r"(\d+)(?:\.\d+)?\.pt$", os.path.basename(path))
        step = int(step_match.group(1)) if step_match else None
        reached = [t["reached_target"] for t in tele if t["reached_target"] is not None]
        results[name] = {
            "checkpoint": path, "algo": algo, "task_phase": task_phase,
            "obs_width": width, "global_step": step, "episodes": int(arr.size),
            "reward_source": rsrc, "reward_kwargs": rkw,
            "checkpoint_env_version": rinfo["env_version"],
            "mean": round(float(arr.mean()), 2),
            "median": round(float(np.median(arr)), 2),
            "std": round(float(arr.std()), 2),
            "min": round(float(arr.min()), 2), "max": round(float(arr.max()), 2),
            "falls_per_episode": round(float(np.mean(falls)), 2),
            "standing_at_end_pct": round(100.0 * np.mean(standing), 1),
            "mean_episode_steps": round(float(np.mean([t["steps"] for t in tele])), 1),
            "reached_target_pct": (round(100.0 * np.mean(reached), 1) if reached else None),
            "mean_min_target_distance": (
                round(float(np.mean([t["min_target_distance"] for t in tele])), 3)
                if reached else None),
            "mean_x_velocity": (
                round(float(np.mean([t["mean_x_velocity"] for t in tele])), 4) if reached
                else None),
        }
        per_episode[name] = [float(r) for r in rewards]
        per_episode[f"{name}__telemetry"] = tele
        print(f"{name:22} mean={arr.mean():10.2f} median={np.median(arr):10.2f} "
              f"std={arr.std():8.2f} min={arr.min():10.2f} max={arr.max():10.2f} "
              f"quedas/ep={np.mean(falls):.2f} de pe={100*np.mean(standing):.0f}% "
              f"alvo={results[name]['reached_target_pct']}% passos="
              f"{results[name]['mean_episode_steps']:.0f} pesos={rsrc}")

    os.makedirs(os.path.dirname(os.path.join(os.getcwd(), args.out)), exist_ok=True)
    payload = {
        "protocol": (f"eval_phase1.py --num-episodes {args.num_episodes} --seed {args.seed} "
                     f"(WalkerRagdoll-v0, reset_mode={args.reset_mode}, "
                     f"one env.step per action, checkpoint obs_rms applied, deterministic; "
                     "same protocol as the Phase-3 table)"),
        "env_version": sys.modules["envs.walker_ragdoll_env"].ENV_VERSION,
        "env_commit": args.env_commit,
        "models": results,
        "per_episode": per_episode,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
