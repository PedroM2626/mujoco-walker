"""
SAC Walker Ragdoll Training with CleanRL-style checkpointing.
"""

import argparse
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import envs.walker_ragdoll_env
from utils.checkpoint import force_delete_run, get_checkpoint_dir, get_run_dir


ENV_VARS = {
    "RUN_ID": "walker_sac",
    "SEED": "1",
    "TOTAL_TIMESTEPS": "1000000",
    "LEARNING_RATE": "3e-4",
    "NUM_ENVS": "16",
    "BUFFER_SIZE": "1000000",
    "GAMMA": "0.99",
    "TAU": "0.005",
    "BATCH_SIZE": "256",
    "LEARNING_STARTS": "10000",
    "POLICY_FREQUENCY": "2",
    "TARGET_NETWORK_FREQUENCY": "1",
    "CHECKPOINT_INTERVAL": "1000000",
    "ALPHA": "0.2",
}


def get_env_or_default(key, default):
    return os.environ.get(key, default)


def parse_args():
    parser = argparse.ArgumentParser(description="SAC Walker Ragdoll Training")
    parser.add_argument("--run-id", type=str, default=get_env_or_default("RUN_ID", ENV_VARS["RUN_ID"]))
    parser.add_argument("--seed", type=int, default=int(get_env_or_default("SEED", ENV_VARS["SEED"])))
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--total-timesteps", type=int, default=int(get_env_or_default("TOTAL_TIMESTEPS", ENV_VARS["TOTAL_TIMESTEPS"])))
    parser.add_argument("--learning-rate", type=float, default=float(get_env_or_default("LEARNING_RATE", ENV_VARS["LEARNING_RATE"])))
    parser.add_argument("--num-envs", type=int, default=int(get_env_or_default("NUM_ENVS", ENV_VARS["NUM_ENVS"])))
    parser.add_argument("--buffer-size", type=int, default=int(get_env_or_default("BUFFER_SIZE", ENV_VARS["BUFFER_SIZE"])))
    parser.add_argument("--gamma", type=float, default=float(get_env_or_default("GAMMA", ENV_VARS["GAMMA"])))
    parser.add_argument("--tau", type=float, default=float(get_env_or_default("TAU", ENV_VARS["TAU"])))
    parser.add_argument("--batch-size", type=int, default=int(get_env_or_default("BATCH_SIZE", ENV_VARS["BATCH_SIZE"])))
    parser.add_argument("--learning-starts", type=int, default=int(get_env_or_default("LEARNING_STARTS", ENV_VARS["LEARNING_STARTS"])))
    parser.add_argument("--policy-frequency", type=int, default=int(get_env_or_default("POLICY_FREQUENCY", ENV_VARS["POLICY_FREQUENCY"])))
    parser.add_argument("--target-network-frequency", type=int, default=int(get_env_or_default("TARGET_NETWORK_FREQUENCY", ENV_VARS["TARGET_NETWORK_FREQUENCY"])))
    parser.add_argument("--checkpoint-interval", type=int, default=int(get_env_or_default("CHECKPOINT_INTERVAL", ENV_VARS["CHECKPOINT_INTERVAL"])))
    parser.add_argument("--alpha", type=float, default=float(get_env_or_default("ALPHA", ENV_VARS["ALPHA"])))
    parser.add_argument("--autotune", action="store_true", default=True)
    parser.add_argument("--no-autotune", dest="autotune", action="store_false")
    parser.add_argument("--capture-video", action="store_true", default=False)
    return parser.parse_args()


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        return env

    return thunk


def layer_init(layer):
    nn.init.xavier_uniform_(layer.weight)
    nn.init.constant_(layer.bias, 0.0)
    return layer


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim + action_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1)),
        )

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=1))


class SACAgent(nn.Module):
    def __init__(self, obs_dim, action_space):
        super().__init__()
        self.action_dim = int(np.prod(action_space.shape))
        self.register_buffer("action_scale", torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32))
        self.backbone = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
        )
        self.fc_mean = layer_init(nn.Linear(256, self.action_dim))
        self.fc_logstd = layer_init(nn.Linear(256, self.action_dim))

    def forward(self, obs):
        x = self.backbone(obs)
        mean = self.fc_mean(x)
        log_std = torch.tanh(self.fc_logstd(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, obs, deterministic=False):
        mean, log_std = self(obs)
        if deterministic:
            y_t = torch.tanh(mean)
            return y_t * self.action_scale + self.action_bias, None, mean

        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, mean


@dataclass
class ReplayBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(self, size, obs_shape, action_shape, device):
        self.size = size
        self.device = device
        self.obs = np.zeros((size, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((size, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((size, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((size, 1), dtype=np.float32)
        self.dones = np.zeros((size, 1), dtype=np.float32)
        self.pos = 0
        self.full = False

    def add(self, obs, next_obs, actions, rewards, dones):
        n = obs.shape[0]
        idxs = (np.arange(n) + self.pos) % self.size
        self.obs[idxs] = obs
        self.next_obs[idxs] = next_obs
        self.actions[idxs] = actions
        self.rewards[idxs, 0] = rewards
        self.dones[idxs, 0] = dones
        self.pos = (self.pos + n) % self.size
        self.full = self.full or self.pos == 0

    def sample(self, batch_size):
        max_idx = self.size if self.full else self.pos
        idxs = np.random.randint(0, max_idx, size=batch_size)
        return ReplayBatch(
            obs=torch.as_tensor(self.obs[idxs], device=self.device),
            actions=torch.as_tensor(self.actions[idxs], device=self.device),
            rewards=torch.as_tensor(self.rewards[idxs], device=self.device),
            next_obs=torch.as_tensor(self.next_obs[idxs], device=self.device),
            dones=torch.as_tensor(self.dones[idxs], device=self.device),
        )


def serialize_replay_buffer(rb: ReplayBuffer) -> Dict[str, Any]:
    return {
        "size": rb.size,
        "pos": rb.pos,
        "full": rb.full,
        "obs": rb.obs,
        "next_obs": rb.next_obs,
        "actions": rb.actions,
        "rewards": rb.rewards,
        "dones": rb.dones,
    }


def restore_replay_buffer(rb: ReplayBuffer, state: Dict[str, Any]):
    if state["size"] != rb.size:
        raise ValueError(
            f"Replay buffer size mismatch: checkpoint={state['size']} current={rb.size}"
        )
    rb.pos = int(state["pos"])
    rb.full = bool(state["full"])
    rb.obs[...] = state["obs"]
    rb.next_obs[...] = state["next_obs"]
    rb.actions[...] = state["actions"]
    rb.rewards[...] = state["rewards"]
    rb.dones[...] = state["dones"]


def get_rng_state() -> Dict[str, Any]:
    return {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_random": torch.get_rng_state(),
        "torch_cuda_random": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
    }


def set_rng_state(state: Dict[str, Any]):
    random.setstate(state["python_random"])
    np.random.set_state(state["numpy_random"])
    torch.set_rng_state(state["torch_random"])
    if torch.cuda.is_available() and state.get("torch_cuda_random") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda_random"])


def reset_envs_without_obs_rms_update(envs, seed):
    """Reset the vector env while preserving restored obs_rms on the first resume step."""
    normalize_env = envs.env
    raw_env = normalize_env.env
    raw_obs, info = raw_env.reset(seed=seed)
    normalized_obs = (raw_obs - normalize_env.obs_rms.mean) / np.sqrt(
        normalize_env.obs_rms.var + normalize_env.epsilon
    )
    return np.clip(normalized_obs, -10, 10), info


def save_sac_checkpoint(
    path,
    global_step,
    actor,
    qf1,
    qf2,
    qf1_target,
    qf2_target,
    actor_optimizer,
    q_optimizer,
    alpha_optimizer,
    log_alpha,
    envs,
    replay_buffer,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "algo": "sac",
            "global_step": global_step,
            "num_envs": envs.num_envs,
            "buffer_size": replay_buffer.size,
            "actor_state_dict": actor.state_dict(),
            "qf1_state_dict": qf1.state_dict(),
            "qf2_state_dict": qf2.state_dict(),
            "qf1_target_state_dict": qf1_target.state_dict(),
            "qf2_target_state_dict": qf2_target.state_dict(),
            "actor_optimizer_state_dict": actor_optimizer.state_dict(),
            "q_optimizer_state_dict": q_optimizer.state_dict(),
            "alpha_optimizer_state_dict": alpha_optimizer.state_dict() if alpha_optimizer is not None else None,
            "log_alpha": log_alpha.detach().cpu(),
            "obs_rms": getattr(envs, "obs_rms", None),
            "replay_buffer": serialize_replay_buffer(replay_buffer),
            "rng_state": get_rng_state(),
        },
        path,
    )
    print(f"[CHECKPOINT] Saved at step {global_step} -> {path}")


def latest_sac_checkpoint(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        return None
    candidates = []
    for name in os.listdir(ckpt_dir):
        if name.startswith("sac_ckpt_") and name.endswith(".pt"):
            try:
                candidates.append((int(name[len("sac_ckpt_") : -3]), os.path.join(ckpt_dir, name)))
            except ValueError:
                pass
    return max(candidates)[1] if candidates else None


def train(start_time=None):
    if start_time is None:
        start_time = time.time()
    args = parse_args()
    run_name = f"{args.run_id}__{args.seed}__{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    if not args.resume or args.force:
        force_delete_run(args.run_id)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    writer = SummaryWriter(get_run_dir(run_name))
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()))

    env_id = "WalkerRagdoll-v0"
    envs = gym.vector.SyncVectorEnv([make_env(env_id, i, args.capture_video, run_name) for i in range(args.num_envs)])
    envs = gym.wrappers.NormalizeObservation(envs)
    envs = gym.wrappers.TransformObservation(envs, lambda obs: np.clip(obs, -10, 10))

    obs_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(np.prod(envs.single_action_space.shape))
    actor = SACAgent(obs_dim, envs.single_action_space).to(device)
    qf1 = SoftQNetwork(obs_dim, action_dim).to(device)
    qf2 = SoftQNetwork(obs_dim, action_dim).to(device)
    qf1_target = SoftQNetwork(obs_dim, action_dim).to(device)
    qf2_target = SoftQNetwork(obs_dim, action_dim).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    actor_optimizer = optim.Adam(actor.parameters(), lr=args.learning_rate)
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
    target_entropy = -float(action_dim)
    log_alpha = torch.tensor(np.log(args.alpha), requires_grad=True, device=device)
    alpha_optimizer = optim.Adam([log_alpha], lr=args.learning_rate) if args.autotune else None

    rb = ReplayBuffer(args.buffer_size, envs.single_observation_space.shape, envs.single_action_space.shape, device)
    global_step = 0
    restored_obs_rms = None

    ckpt_dir = get_checkpoint_dir(args.run_id)
    if args.resume:
        ckpt_path = latest_sac_checkpoint(ckpt_dir)
        if ckpt_path:
            print(f"[CHECKPOINT] Loading from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)
            if checkpoint.get("algo") != "sac":
                raise ValueError(f"Checkpoint {ckpt_path} is not a SAC checkpoint.")
            if checkpoint.get("buffer_size") is not None and int(checkpoint["buffer_size"]) != args.buffer_size:
                raise ValueError(
                    f"Replay buffer size mismatch for resume: checkpoint={checkpoint['buffer_size']} current={args.buffer_size}"
                )
            actor.load_state_dict(checkpoint["actor_state_dict"])
            qf1.load_state_dict(checkpoint["qf1_state_dict"])
            qf2.load_state_dict(checkpoint["qf2_state_dict"])
            qf1_target.load_state_dict(checkpoint["qf1_target_state_dict"])
            qf2_target.load_state_dict(checkpoint["qf2_target_state_dict"])
            actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
            q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
            log_alpha = checkpoint["log_alpha"].to(device).requires_grad_()
            alpha_optimizer = optim.Adam([log_alpha], lr=args.learning_rate) if args.autotune else None
            if alpha_optimizer is not None and checkpoint.get("alpha_optimizer_state_dict") is not None:
                alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
            if checkpoint.get("replay_buffer") is not None:
                restore_replay_buffer(rb, checkpoint["replay_buffer"])
            if checkpoint.get("obs_rms") is not None:
                envs.obs_rms = checkpoint["obs_rms"]
                restored_obs_rms = checkpoint["obs_rms"]
            if checkpoint.get("rng_state") is not None:
                set_rng_state(checkpoint["rng_state"])
            global_step = int(checkpoint.get("global_step", 0))

    if restored_obs_rms is not None:
        next_obs, _ = reset_envs_without_obs_rms_update(envs, seed=args.seed)
    else:
        next_obs, _ = envs.reset(seed=args.seed)
    next_obs = next_obs.astype(np.float32)
    next_checkpoint_step = ((global_step // args.checkpoint_interval) + 1) * args.checkpoint_interval

    try:
        while global_step < args.total_timesteps:
            obs = next_obs
            if global_step < args.learning_starts:
                actions = np.array([envs.single_action_space.sample() for _ in range(args.num_envs)])
            else:
                with torch.no_grad():
                    actions, _, _ = actor.get_action(torch.as_tensor(obs, dtype=torch.float32, device=device))
                    actions = actions.cpu().numpy()

            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            next_obs = next_obs.astype(np.float32)
            dones = np.logical_or(terminations, truncations).astype(np.float32)
            rb.add(obs, next_obs, actions, rewards.astype(np.float32), dones)
            global_step += args.num_envs

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        ep_r = float(info["episode"]["r"])
                        ep_l = float(info["episode"]["l"])
                        print(f"global_step={global_step}, episodic_return={ep_r:.2f}, episodic_length={ep_l:.0f}")
                        writer.add_scalar("charts/episodic_return", ep_r, global_step)
                        writer.add_scalar("charts/episodic_length", ep_l, global_step)

            if global_step > args.learning_starts:
                batch = rb.sample(args.batch_size)
                with torch.no_grad():
                    next_actions, next_log_pi, _ = actor.get_action(batch.next_obs)
                    qf1_next_target = qf1_target(batch.next_obs, next_actions)
                    qf2_next_target = qf2_target(batch.next_obs, next_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - log_alpha.exp() * next_log_pi
                    next_q_value = batch.rewards + (1 - batch.dones) * args.gamma * min_qf_next_target

                qf1_a_values = qf1(batch.obs, batch.actions)
                qf2_a_values = qf2(batch.obs, batch.actions)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                if global_step % (args.policy_frequency * args.num_envs) == 0:
                    for _ in range(args.policy_frequency):
                        pi, log_pi, _ = actor.get_action(batch.obs)
                        qf1_pi = qf1(batch.obs, pi)
                        qf2_pi = qf2(batch.obs, pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)
                        actor_loss = ((log_alpha.exp() * log_pi) - min_qf_pi).mean()

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

                if global_step % (args.target_network_frequency * args.num_envs) == 0:
                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                if global_step % 1000 < args.num_envs:
                    writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                    writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                    writer.add_scalar("losses/qf_loss", qf_loss.item(), global_step)
                    writer.add_scalar("losses/alpha", log_alpha.exp().item(), global_step)
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            if global_step >= next_checkpoint_step:
                save_sac_checkpoint(
                    os.path.join(ckpt_dir, f"sac_ckpt_{global_step}.pt"),
                    global_step,
                    actor,
                    qf1,
                    qf2,
                    qf1_target,
                    qf2_target,
                    actor_optimizer,
                    q_optimizer,
                    alpha_optimizer,
                    log_alpha,
                    envs,
                    rb,
                )
                next_checkpoint_step += args.checkpoint_interval
    except KeyboardInterrupt:
        print("\n[TRAIN] Interrupted by user. Saving checkpoint...")
    finally:
        save_sac_checkpoint(
            os.path.join(ckpt_dir, f"sac_ckpt_{global_step}.pt"),
            global_step,
            actor,
            qf1,
            qf2,
            qf1_target,
            qf2_target,
            actor_optimizer,
            q_optimizer,
            alpha_optimizer,
            log_alpha,
            envs,
            rb,
        )
        envs.close()
        writer.close()
        print(f"Training completed. Total steps: {global_step}")


if __name__ == "__main__":
    train(start_time=time.time())
