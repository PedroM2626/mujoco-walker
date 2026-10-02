"""
Visualize a trained Walker Ragdoll policy in real time.

Three ways in:
    python play.py --run-id walker_target_v1                 # latest checkpoint of a run
    python play.py --checkpoint merged_avg_model.pt          # any .pt, incl. root-level merges
    python play.py --moe --gate moe_gate.pt \
                   --recovery checkpoints/walker_recovery_v1/sac_ckpt_20000000.pt \
                   --target checkpoints/walker_target_v1/sac_ckpt_40000000.pt
"""

import argparse
import os
import sys
import time

import gymnasium as gym
import mujoco.viewer
import numpy as np
import torch

import envs.walker_ragdoll_env
from envs.reward_shaping import reward_kwargs_for
from train_walker import (
    SACAgent, PPOAgent,
    latest_checkpoint_any,
    load_torch_checkpoint,
    RecoverySupervisor,
    wrap_clip_action,
)
from train_moe_gate import MoEGate
from utils.checkpoint import get_checkpoint_dir
EPSILON = 1e-8
CLIP = 10.0


def policy_input(obs, obs_rms, dim, device):
    """Slice to the policy's observation width, then apply that policy's own statistics.

    The recovery expert takes 46 dims and the target expert 49 while rollouts always
    produce 49, so each branch must be cut and normalised by its own stats.
    """
    array = np.asarray(obs, dtype=np.float64)
    if array.shape[0] > dim:
        array = array[:dim]
    if obs_rms is not None:
        mean = np.asarray(obs_rms.mean, dtype=np.float64)
        var = np.asarray(obs_rms.var, dtype=np.float64)
        if mean.shape[0] != array.shape[0]:
            raise ValueError(
                f"obs_rms has {mean.shape[0]} dims but the policy takes {array.shape[0]}"
            )
        array = np.clip((array - mean) / np.sqrt(var + EPSILON), -CLIP, CLIP)
    return torch.FloatTensor(array).unsqueeze(0).to(device)


def parse_args():
    parser = argparse.ArgumentParser(description="Play a trained SAC / PPO / merged / MoE Walker Ragdoll agent")
    parser.add_argument("--run-id", type=str, default=None, help="Run name under checkpoints/.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Direct path to a .pt (lets you load the root-level merged models).")
    parser.add_argument("--moe", action="store_true", default=False,
                        help="Mixture-of-experts mode: route between --recovery and --target with --gate.")
    parser.add_argument("--gate", type=str, default="moe_gate.pt")
    parser.add_argument("--recovery", type=str, default=None, help="Recovery expert checkpoint (46-dim).")
    parser.add_argument("--target", type=str, default=None, help="Walking/target expert checkpoint (49-dim).")
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--num-episodes", type=int, default=0, help="0 = infinite until Ctrl+C")
    parser.add_argument("--render-mode", type=str, default="human", choices=["human", "rgb_array"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--stochastic", action="store_true", default=False, help="Sample from the policy instead of using the mean action")
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument(
        "--reset-mode",
        type=str,
        default="mixed",
        choices=["fixed", "mixed", "fallen", "upright"],
        help="Initial-state distribution used while visualizing.",
    )
    parser.add_argument(
        "--task-phase",
        type=str,
        default=None,
        choices=["recovery", "balance", "walk", "target"],
        help="Reward phase used while visualizing. Defaults to the checkpoint phase when available.",
    )
    parser.add_argument("--use-supervisor", action="store_true", default=False, help="Use the Recovery Supervisor when fallen and show visualization")
    args = parser.parse_args()

    if args.moe:
        missing = [n for n in (args.gate, args.recovery, args.target) if not n or not os.path.exists(n)]
        if missing:
            parser.error(f"--moe needs existing --gate/--recovery/--target (missing: {', '.join(missing)})")
    elif not args.run_id and not args.checkpoint:
        parser.error("give --run-id, --checkpoint, or --moe with --gate/--recovery/--target")
    return args


def make_base_env(env_id, reset_mode="mixed", task_phase="recovery", reward_kwargs=None):
    # reward_kwargs must come from the checkpoint being played: the trainer shapes the reward and
    # the environment defaults do not match it, so the dashboard the user watches would otherwise
    # report a return for a reward function the policy never optimised.
    env = gym.make(
        env_id,
        render_mode="rgb_array",
        reset_mode=reset_mode,
        task_phase=task_phase,
        **(reward_kwargs or {}),
    )
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = wrap_clip_action(env)
    return env


def resolve_checkpoint(run_id, checkpoint_step):
    ckpt_dir = get_checkpoint_dir(run_id)
    if checkpoint_step is not None:
        # Try SAC first, then PPO
        for prefix in ("sac_ckpt_", "ppo_ckpt_"):
            ckpt_path = os.path.join(ckpt_dir, f"{prefix}{checkpoint_step}.pt")
            if os.path.exists(ckpt_path):
                return ckpt_path
        print(f"[ERROR] Checkpoint for step {checkpoint_step} not found. Falling back to latest.")
    return latest_checkpoint_any(ckpt_dir)


def load_single(args, device):
    """Load one policy from --checkpoint or --run-id. Returns (env, act_fn, label, task_phase)."""
    if args.checkpoint:
        if not os.path.exists(args.checkpoint):
            print(f"[ERROR] No such checkpoint: {args.checkpoint}")
            sys.exit(1)
        ckpt_path = args.checkpoint
    else:
        ckpt_path = resolve_checkpoint(args.run_id, args.checkpoint_step)
        if ckpt_path is None:
            print(f"[ERROR] No checkpoint found for run-id '{args.run_id}'.")
            sys.exit(1)

    checkpoint = load_torch_checkpoint(ckpt_path, device)
    algo = checkpoint.get("algo", "unknown")
    if algo not in {"sac", "sac_actor", "ppo"}:
        print(f"[ERROR] {ckpt_path} has unknown algorithm type '{algo}'.")
        sys.exit(1)

    obs_rms = checkpoint.get("obs_rms")
    if obs_rms is None:
        # Merged checkpoints may not carry statistics. Running raw is a legitimate
        # choice to inspect, but it must be loud: an un-normalised policy is a
        # different policy from the one that was trained.
        print(f"[WARN] {ckpt_path} has no obs_rms -> feeding RAW observations.")

    task_phase = args.task_phase or checkpoint.get("task_phase") or "recovery"
    state_dict = checkpoint.get("actor_state_dict", checkpoint)

    # The checkpoint's own input width is the only trustworthy source: merged models carry
    # no task_phase, and building a 46-wide env for a 49-wide policy fails deep inside
    # load_state_dict with a size mismatch.
    policy_width = int(state_dict["backbone.0.weight"].shape[1])
    if policy_width == 49 and task_phase != "target":
        print(f"[PLAY] checkpoint policy takes {policy_width} dims -> forcing task_phase='target'")
        task_phase = "target"

    rkw, rsrc = reward_kwargs_for(checkpoint)
    print(f"[PLAY] recompensa usada: {rsrc}")
    env = make_base_env("WalkerRagdoll-v0", reset_mode=args.reset_mode, task_phase=task_phase,
                        reward_kwargs=rkw)
    obs_dim = int(np.prod(env.observation_space.shape))
    if obs_dim != policy_width:
        print(f"[ERROR] checkpoint policy expects {policy_width} observations but "
              f"{task_phase!r} phase produces {obs_dim}. Pass --task-phase explicitly.")
        sys.exit(1)

    if algo in {"sac", "sac_actor"}:
        agent = SACAgent(obs_dim, env.action_space).to(device)
        agent.load_state_dict(checkpoint["actor_state_dict"])
        agent.eval()
        label = "SAC"

        def act(obs):
            tensor = policy_input(obs, obs_rms, obs_dim, device)
            action, _, _ = agent.get_action(tensor, deterministic=not args.stochastic)
            return action[0]
    else:
        action_dim = int(np.prod(env.action_space.shape))
        agent = PPOAgent(obs_dim, action_dim).to(device)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        agent.eval()
        label = "PPO"

        def act(obs):
            tensor = policy_input(obs, obs_rms, obs_dim, device)
            if args.stochastic:
                a, _, _, _ = agent.get_action_and_value(tensor)
            else:
                a = agent.get_deterministic_action(tensor)
            return a[0]

    return env, act, label, task_phase


def load_moe(args, device):
    """Recreate the evaluate_merging MoE router: g*recovery + (1-g)*walking."""
    rec_ckpt = load_torch_checkpoint(args.recovery, device)
    tgt_ckpt = load_torch_checkpoint(args.target, device)
    # The walking expert decides the scoring reward here, because this env is built for the
    # target-phase rollouts the router is evaluated on.
    env = make_base_env(
        "WalkerRagdoll-v0",
        reset_mode=args.reset_mode,
        task_phase=args.task_phase or "target",
        reward_kwargs=reward_kwargs_for(tgt_ckpt)[0],
    )
    print(f"[PLAY MoE] recompensa usada: {reward_kwargs_for(tgt_ckpt)[1]}")
    rec = SACAgent(46, env.action_space).to(device)
    rec.load_state_dict(rec_ckpt.get("actor_state_dict", rec_ckpt))
    rec.eval()
    tgt = SACAgent(49, env.action_space).to(device)
    tgt.load_state_dict(tgt_ckpt.get("actor_state_dict", tgt_ckpt))
    tgt.eval()
    gate = MoEGate().to(device)
    gate.load_state_dict(torch.load(args.gate, map_location=device, weights_only=True))
    gate.eval()
    rec_rms, tgt_rms = rec_ckpt.get("obs_rms"), tgt_ckpt.get("obs_rms")
    if rec_rms is None or tgt_rms is None:
        print("[WARN] an expert has no obs_rms; that branch sees RAW observations.")

    def act(obs):
        rec_in = policy_input(obs, rec_rms, 46, device)
        tgt_in = policy_input(obs, tgt_rms, 49, device)
        g = float(gate(rec_in).item())
        a_rec = rec.get_action(rec_in, deterministic=True)[0]
        a_tgt = tgt.get_action(tgt_in, deterministic=True)[0]
        return (g * a_rec + (1.0 - g) * a_tgt)[0]

    return env, act, f"MoE (gate={os.path.basename(args.gate)})", env.unwrapped._task_phase


def play():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.moe:
        env, act, label, task_phase = load_moe(args, device)
    else:
        env, act, label, task_phase = load_single(args, device)

    print(f"[PLAY] policy={label} task_phase={task_phase}")
    print(f"[PLAY] {'Stochastic' if args.stochastic else 'Deterministic'} policy")

    obs, _ = env.reset(seed=args.seed)

    viewer = None
    if args.render_mode == "human":
        viewer = mujoco.viewer.launch_passive(env.unwrapped.model, env.unwrapped.data)

    supervisor = None
    if args.use_supervisor:
        print("[PLAY] Initializing Recovery Supervisor...")
        supervisor = RecoverySupervisor(device)

    total_reward = 0.0
    episode_count = 0
    step_count = 0
    infinite = args.num_episodes <= 0

    try:
        while True if infinite else episode_count < args.num_episodes:
            with torch.no_grad():
                action = act(obs)
                action = action.cpu().numpy() if torch.is_tensor(action) else np.asarray(action)

            if supervisor is not None:
                actions_array = np.array([action])
                actions_array, is_rec = supervisor.get_actions([env], actions_array)
                action = actions_array[0]

                # Visual feedback: Change walker color when recovery network is active
                if viewer is not None:
                    model = env.unwrapped.model
                    for geom_id in range(model.ngeom):
                        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                        if name not in ["floor", "target_marker_geom"]:
                            if is_rec[0]:
                                model.geom_rgba[geom_id] = [1.0, 0.0, 0.0, 1.0] # Red
                            else:
                                model.geom_rgba[geom_id] = [0.8, 0.6, 0.4, 1.0] # Original color approx

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            if args.render_mode == "human" and viewer is not None:
                viewer.sync()
                if not viewer.is_running():
                    print("[PLAY] Viewer closed, exiting.")
                    break
                time.sleep(1.0 / args.fps)
            # rgb_array used to call env.render() and throw the frame away: a full
            # 480x480 offscreen render per step with no consumer. "rgb_array" here now
            # means headless replay; use --capture-video in the trainers for video.

            if terminated or truncated:
                episode_count += 1
                ep_return = info.get("episode", {}).get("r", total_reward)
                ep_length = info.get("episode", {}).get("l", step_count)
                ep_return = ep_return.item() if hasattr(ep_return, "item") else float(ep_return)
                ep_length = ep_length.item() if hasattr(ep_length, "item") else int(ep_length)
                print(f"[EPISODE {episode_count}/{args.num_episodes}] Return={ep_return:.2f}  Length={ep_length}")
                total_reward = 0.0
                step_count = 0
                obs, _ = env.reset()
    except KeyboardInterrupt:
        print("\n[PLAY] Interrupted by user.")
    finally:
        env.close()
        if viewer is not None:
            viewer.close()
        print("[PLAY] Finished.")


if __name__ == "__main__":
    play()
