import os
import random
import time
import argparse
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import mlflow
from torch.utils.tensorboard import SummaryWriter

# Import utilities and environment from train_walker
from train_walker import (
    make_env,
    add_vec_env_args,
    build_vec_env,
    add_device_arg,
    select_device,
    SACAgent,
    SoftQNetwork,
    BatchedSoftQEnsemble,
    ReplayBuffer,
    force_delete_run,
    get_checkpoint_dir,
    load_torch_checkpoint,
    restore_replay_buffer,
    adapt_obs_rms,
    set_obs_rms,
    wrap_normalize_observation,
    wrap_transform_observation,
    get_rng_state,
    set_rng_state,
    ENV_VERSION,
    start_mlflow_run,
    log_mlflow_metrics,
    log_mlflow_artifact,
    end_mlflow_run,
)
from envs.reward_shaping import TRAINING_REWARD_KWARGS

def make_ensemble(obs_dim, action_dim, size, impl, device):
    """REDQ's N critics: the original nn.ModuleList, or one batched bmm stack.

    Same weights, same numbers - the forward and all 3*N weight gradients agree to float32
    reduction order, which is what tests/test_redq_ensemble.py pins. The reason to switch is that
    the loop's cost is kernel launches, not arithmetic: `python bench_redq_ensemble.py
    --isolated-only` times one critic step and one target soft update for both layouts and writes
    them to benchmarks/redq_ensemble_ab.json under "isolated".
    """
    if impl == "batched":
        return BatchedSoftQEnsemble(obs_dim, action_dim, size).to(device)
    return nn.ModuleList([SoftQNetwork(obs_dim, action_dim) for _ in range(size)]).to(device)


def ensemble_q(ens, obs, action):
    """Every critic's value for a batch, shaped (N, B, 1) either way."""
    if isinstance(ens, BatchedSoftQEnsemble):
        return ens(obs, action)
    return torch.stack([q(obs, action) for q in ens], dim=0)


def soft_update_ensemble(target, source, tau):
    if isinstance(target, BatchedSoftQEnsemble):
        target.lerp_from(source, tau)
        return
    for q_net, t_net in zip(source, target):
        for p, tp in zip(q_net.parameters(), t_net.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)


def ensemble_state_dict(ens):
    """Checkpoints are always written in the nn.ModuleList layout, whichever impl ran."""
    return ens.as_module_list_state_dict() if isinstance(ens, BatchedSoftQEnsemble) else ens.state_dict()


def load_ensemble_state(ens, state):
    if isinstance(ens, BatchedSoftQEnsemble):
        ens.load_module_list_state_dict(state)
    else:
        ens.load_state_dict(state)


def parse_redq_args():
    parser = argparse.ArgumentParser(description="REDQ Walker Ragdoll Training")
    parser.add_argument("--run-id", type=str, default="walker_redq_1m")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--total-timesteps", type=int, default=1000000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-starts", type=int, default=5000)
    parser.add_argument("--policy-frequency", type=int, default=2)
    parser.add_argument("--checkpoint-interval", type=int, default=200000)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--autotune", action="store_true", default=False)
    parser.add_argument("--no-autotune", dest="autotune", action="store_false")
    parser.set_defaults(autotune=True)
    parser.add_argument("--reset-mode", type=str, default="upright")
    parser.add_argument("--fixed-reset-probability", type=float, default=0.25)
    parser.add_argument("--upright-reset-probability", type=float, default=0.15)
    parser.add_argument("--fallen-velocity-scale", type=float, default=0.35)
    parser.add_argument("--task-phase", type=str, default="target")
    parser.add_argument("--target-forward-velocity", type=float, default=0.8)
    parser.add_argument("--init-from-run-id", type=str, default=None)
    parser.add_argument("--init-from-checkpoint-step", type=int, default=0)
    parser.add_argument("--allow-mismatched-env-version", action="store_true", default=False)
    parser.add_argument("--save-replay-buffer", action="store_true", default=False)
    add_vec_env_args(parser)
    add_device_arg(parser)
    
    # REDQ specific arguments
    parser.add_argument("--ensemble-size", type=int, default=10, help="Number of Q-networks in ensemble (N)")
    parser.add_argument("--num-min-critics", type=int, default=2, help="Number of Q-networks sampled for target (M)")
    parser.add_argument("--utd-ratio", type=int, default=20,  help="Updates to Data ratio (G)")
    parser.add_argument("--ensemble-impl", type=str, default="batched", choices=["batched", "loop"],
                        help="batched = one bmm over the N critics (default; measured equivalent to "
                             "the loop and ~8x faster per critic step on the GPU). loop = the "
                             "original nn.ModuleList, kept for A/B runs against old numbers.")

    return parser.parse_args()

def save_redq_checkpoint(
    ckpt_path,
    global_step,
    actor,
    q_ensemble,
    target_ensemble,
    actor_optimizer,
    q_optimizer,
    alpha_optimizer,
    log_alpha,
    envs,
    rb,
    task_phase,
    target_forward_velocity,
    save_replay_buffer,
    ensemble_size,
    num_min_critics,
    utd_ratio,
):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    # Save the lightweight actor weights separately
    actor_only_path = ckpt_path.replace("redq_ckpt_", "redq_actor_")
    torch.save(actor.state_dict(), actor_only_path)
    
    # Save full state
    state = {
        "algo": "redq",
        "env_version": ENV_VERSION,
        "global_step": global_step,
        "actor_state_dict": actor.state_dict(),
        "q_ensemble_state_dict": ensemble_state_dict(q_ensemble),
        "target_ensemble_state_dict": ensemble_state_dict(target_ensemble),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "q_optimizer_state_dict": q_optimizer.state_dict(),
        "log_alpha": log_alpha.detach().cpu(),
        "obs_rms": envs.obs_rms if hasattr(envs, "obs_rms") else None,
        "task_phase": task_phase,
        "target_forward_velocity": target_forward_velocity,
        # Recorded so an evaluator scores this policy under the reward it was trained with. Without
        # it the scorer has to guess, and it guessed wrong once already (see reward_kwargs_for).
        "reward_kwargs": dict(TRAINING_REWARD_KWARGS),
        "rng_state": get_rng_state(),
        "ensemble_size": ensemble_size,
        "num_min_critics": num_min_critics,
        "utd_ratio": utd_ratio,
        # Adam's state is per parameter, and the two ensemble layouts have a different number of
        # parameters (6 stacks vs 3N tensors), so a checkpoint is only resumable by the
        # implementation that wrote it. Record which one that was.
        "ensemble_impl": "batched" if isinstance(q_ensemble, BatchedSoftQEnsemble) else "loop",
    }
    if alpha_optimizer is not None:
        state["alpha_optimizer_state_dict"] = alpha_optimizer.state_dict()
    if save_replay_buffer:
        state["replay_buffer"] = rb
        state["buffer_size"] = rb.buffer_size

    torch.save(state, ckpt_path)
    print(f"[CHECKPOINT] Saved at step {global_step} -> {ckpt_path}")

def train_redq():
    args = parse_redq_args()
    run_name = f"{args.run_id}__{args.seed}"
    start_time = time.time()

    if not args.resume or args.force:
        force_delete_run(args.run_id, run_name)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = select_device(args)
    print(f"Using device: {device}")

    envs = build_vec_env(args, run_name, capture_video=False)
    envs = wrap_normalize_observation(envs)
    envs = wrap_transform_observation(envs, lambda obs: np.clip(obs, -10, 10))

    obs_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(np.prod(envs.single_action_space.shape))

    actor = SACAgent(obs_dim, envs.single_action_space).to(device)
    
    # Initialize REDQ Q-ensemble (N critics, batched or as a list - see --ensemble-impl)
    q_ensemble = make_ensemble(obs_dim, action_dim, args.ensemble_size, args.ensemble_impl, device)
    target_ensemble = make_ensemble(obs_dim, action_dim, args.ensemble_size, args.ensemble_impl, device)
    load_ensemble_state(target_ensemble, ensemble_state_dict(q_ensemble))
    print(f"[REDQ] ensemble: N={args.ensemble_size} impl={args.ensemble_impl}")

    actor_optimizer = optim.Adam(actor.parameters(), lr=args.learning_rate)
    q_optimizer = optim.Adam(q_ensemble.parameters(), lr=args.learning_rate)
    target_entropy = -float(action_dim)
    log_alpha = torch.tensor(np.log(args.alpha), requires_grad=True, device=device)
    alpha_optimizer = optim.Adam([log_alpha], lr=args.learning_rate) if args.autotune else None

    rb = ReplayBuffer(args.buffer_size, envs.single_observation_space.shape, envs.single_action_space.shape, device)
    global_step = 0
    gradient_steps = 0
    restored_obs_rms = None

    ckpt_dir = get_checkpoint_dir(args.run_id)
    
    # Setup WandB (removido: tracking canônico é MLflow + TensorBoard; wandb não é dependência)
    # Setup MLflow
    mlf_run = start_mlflow_run(args, run_name, "redq")

    # Initialize from checkpoint or pre-trained weights
    if args.resume:
        # Find latest checkpoint
        files = [f for f in os.listdir(ckpt_dir) if f.startswith("redq_ckpt_") and f.endswith(".pt")] if os.path.exists(ckpt_dir) else []
        if files:
            files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            ckpt_path = os.path.join(ckpt_dir, files[-1])
            print(f"[CHECKPOINT] Loading from {ckpt_path}")
            checkpoint = load_torch_checkpoint(ckpt_path, device)
            saved_impl = checkpoint.get("ensemble_impl", "loop")
            if saved_impl != args.ensemble_impl:
                raise SystemExit(
                    f"[REDQ] {ckpt_path} foi gravado com --ensemble-impl {saved_impl!r}. O estado "
                    f"do optimizador e por-parametro e as duas implementacoes tem um numero "
                    f" diferente de tensores (6 vs 3N), entao retoma com "
                    f"--ensemble-impl {saved_impl}; para levar apenas os pesos de uma "
                    f"implementacao para a outra usa --init-from-run-id.")
            actor.load_state_dict(checkpoint["actor_state_dict"])
            load_ensemble_state(q_ensemble, checkpoint["q_ensemble_state_dict"])
            load_ensemble_state(target_ensemble, checkpoint["target_ensemble_state_dict"])
            actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
            q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
            log_alpha = checkpoint["log_alpha"].to(device).requires_grad_()
            alpha_optimizer = optim.Adam([log_alpha], lr=args.learning_rate) if args.autotune else None
            if alpha_optimizer is not None and checkpoint.get("alpha_optimizer_state_dict") is not None:
                alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
            if checkpoint.get("replay_buffer") is not None:
                restore_replay_buffer(rb, checkpoint["replay_buffer"])
            if checkpoint.get("obs_rms") is not None:
                restored_obs_rms = adapt_obs_rms(checkpoint["obs_rms"], envs.single_observation_space.shape)
                set_obs_rms(envs, restored_obs_rms)
            if checkpoint.get("rng_state") is not None:
                set_rng_state(checkpoint["rng_state"])
            global_step = checkpoint["global_step"]
    elif args.init_from_run_id:
        init_ckpt_dir = get_checkpoint_dir(args.init_from_run_id)
        if os.path.exists(init_ckpt_dir):
            # Resolve latest checkpoint
            files = [f for f in os.listdir(init_ckpt_dir) if f.endswith(".pt") and ("ckpt_" in f or "actor_" in f)]
            if files:
                files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
                init_ckpt_path = os.path.join(init_ckpt_dir, files[-1])
                print(f"[LOAD] Initializing weights from {init_ckpt_path}")
                checkpoint = load_torch_checkpoint(init_ckpt_path, device)
                
                # Check for actor_state_dict keys
                actor_keys = checkpoint.get("actor_state_dict", checkpoint)
                actor.load_state_dict(actor_keys, strict=False)
                
                if "obs_rms" in checkpoint and checkpoint["obs_rms"] is not None:
                    restored_obs_rms = adapt_obs_rms(checkpoint["obs_rms"], envs.single_observation_space.shape)
                    set_obs_rms(envs, restored_obs_rms)
                    print("[LOAD] Initialized observation normalization statistics.")

    writer = SummaryWriter(os.path.join("runs", run_name))

    obs = envs.reset()[0]
    next_checkpoint_step = ((global_step // args.checkpoint_interval) + 1) * args.checkpoint_interval

    try:
        while global_step < args.total_timesteps:
            # Action selection
            if global_step < args.learning_starts:
                actions = np.array([envs.single_action_space.sample() for _ in range(args.num_envs)])
            else:
                with torch.no_grad():
                    actions, _, _ = actor.get_action(
                        torch.as_tensor(obs, dtype=torch.float32, device=device)
                    )
                    actions = actions.cpu().numpy()

            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            # Record episode returns
            for info in infos.get("final_info", []):
                if info is not None and "episode" in info:
                    eps_ret = float(np.asarray(info["episode"]["r"]).item())
                    eps_len = float(np.asarray(info["episode"]["l"]).item())
                    print(f"global_step={global_step}, episodic_return={eps_ret:.2f}, episodic_length={eps_len:.0f}")
                    writer.add_scalar("charts/episodic_return", eps_ret, global_step)
                    writer.add_scalar("charts/episodic_length", eps_len, global_step)
                    log_mlflow_metrics(mlf_run, {"episodic_return": eps_ret, "episodic_length": eps_len}, global_step)

            # Replay buffer handling
            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]

            dones = np.logical_or(terminations, truncations).astype(np.float32)
            rb.add(obs, real_next_obs, actions, rewards, dones)
            obs = next_obs
            global_step += args.num_envs

            # REDQ Training update steps
            if global_step >= args.learning_starts:
                # Perform G updates per env step (utd_ratio)
                for _ in range(args.utd_ratio):
                    gradient_steps += 1
                    batch = rb.sample(args.batch_size)
                    
                    with torch.no_grad():
                        next_state_actions, next_state_log_pi, _ = actor.get_action(batch.next_obs)
                        
                        # Sample M critics from N ensemble size (REDQ core). Indexing the (N, B, 1)
                        # stack takes the same subset the ModuleList path pulled one critic at a time.
                        sampled_indices = random.sample(range(args.ensemble_size), args.num_min_critics)
                        target_qs = ensemble_q(
                            target_ensemble, batch.next_obs, next_state_actions)[sampled_indices]
                        min_target_q = target_qs.min(dim=0, keepdim=True)[0]

                        alpha = log_alpha.exp()
                        next_q_value = min_target_q - alpha * next_state_log_pi
                        target_q_value = batch.rewards + (1.0 - batch.dones) * args.gamma * next_q_value

                    # Update all Q-networks in the ensemble. Mean over the stack is the same
                    # number as the mean of the per-critic means the loop path summed.
                    qf_loss = F.mse_loss(ensemble_q(q_ensemble, batch.obs, batch.actions),
                                         target_q_value)

                    q_optimizer.zero_grad()
                    qf_loss.backward()
                    q_optimizer.step()

                    # Delayed Actor and Alpha Updates
                    # `global_step` is constant inside the UTD loop, so gating on it
                    # made the condition evaluate the same way for all G iterations:
                    # with policy_frequency=2 and num_envs=16 it was always true and the
                    # actor was updated G times per env step instead of every 2 gradient
                    # steps. Gate on the gradient-step counter instead.
                    if gradient_steps % args.policy_frequency == 0:
                        pi, log_pi, _ = actor.get_action(batch.obs)
                        
                        # Update actor to maximize the mean Q-value of all critics in the ensemble
                        mean_actor_q = ensemble_q(q_ensemble, batch.obs, pi).mean(dim=0, keepdim=True)
                        
                        alpha = log_alpha.exp()
                        actor_loss = ((alpha * log_pi) - mean_actor_q).mean()

                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_optimizer.step()

                        if args.autotune:
                            alpha_loss = (-log_alpha * (log_pi + target_entropy).detach()).mean()
                            alpha_optimizer.zero_grad()
                            alpha_loss.backward()
                            alpha_optimizer.step()
                        else:
                            alpha_loss = torch.tensor(0.0)

                    # Soft update targets for all Q-networks: six kernels on the batched stack,
                    # 3N on the ModuleList. Same tau, same schedule (every gradient step).
                    soft_update_ensemble(target_ensemble, q_ensemble, args.tau)

                # Logging
                if global_step % 1000 < args.num_envs:
                    writer.add_scalar("losses/qf_loss", qf_loss.item(), global_step)
                    writer.add_scalar("losses/alpha", log_alpha.exp().item(), global_step)
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    log_mlflow_metrics(mlf_run, {
                        "qf_loss": qf_loss.item(),
                        "alpha": log_alpha.exp().item(),
                        "sps": int(global_step / (time.time() - start_time)),
                    }, global_step)

            # Checkpoint saving
            if global_step >= next_checkpoint_step:
                ckpt_path = os.path.join(ckpt_dir, f"redq_ckpt_{global_step}.pt")
                save_redq_checkpoint(
                    ckpt_path,
                    global_step,
                    actor,
                    q_ensemble,
                    target_ensemble,
                    actor_optimizer,
                    q_optimizer,
                    alpha_optimizer,
                    log_alpha,
                    envs,
                    rb,
                    args.task_phase,
                    args.target_forward_velocity,
                    args.save_replay_buffer,
                    args.ensemble_size,
                    args.num_min_critics,
                    args.utd_ratio,
                )
                    # Log checkpoint artifact
                log_mlflow_artifact(mlf_run, ckpt_path, "redq", global_step)
                next_checkpoint_step += args.checkpoint_interval

    except KeyboardInterrupt:
        print("\n[TRAIN] Interrupted by user. Saving checkpoint...")
    finally:
        final_checkpoint_path = os.path.join(ckpt_dir, f"redq_ckpt_{global_step}.pt")
        save_redq_checkpoint(
            final_checkpoint_path,
            global_step,
            actor,
            q_ensemble,
            target_ensemble,
            actor_optimizer,
            q_optimizer,
            alpha_optimizer,
            log_alpha,
            envs,
            rb,
            args.task_phase,
            args.target_forward_velocity,
            args.save_replay_buffer,
            args.ensemble_size,
            args.num_min_critics,
            args.utd_ratio,
        )
        log_mlflow_artifact(mlf_run, final_checkpoint_path, "redq", global_step)
        end_mlflow_run(mlf_run)
        envs.close()
        writer.close()
        print(f"Training completed. Total steps: {global_step}")

if __name__ == "__main__":
    train_redq()
