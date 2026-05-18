"""
SAC / PPO Walker Ragdoll Training with CleanRL-style checkpointing.
"""

import argparse
import copy
import gc
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import envs.walker_ragdoll_env
from envs.walker_ragdoll_env import ENV_VERSION
from utils.checkpoint import force_delete_run, get_checkpoint_dir, get_run_dir

try:
    import mlflow
    import mlflow.pytorch
except ImportError:
    mlflow = None


ENV_VARS = {
    "ALGO": "sac",
    "RUN_ID": "walker_train",
    "SEED": "1",
    "TOTAL_TIMESTEPS": "1000000",
    "LEARNING_RATE": "3e-4",
    "NUM_ENVS": "32",
    "BUFFER_SIZE": "1000000",
    "GAMMA": "0.99",
    "TAU": "0.005",
    "BATCH_SIZE": "512",
    "LEARNING_STARTS": "10000",
    "POLICY_FREQUENCY": "2",
    "TARGET_NETWORK_FREQUENCY": "1",
    "CHECKPOINT_INTERVAL": "1000000",
    "ALPHA": "0.2",
    "RESET_MODE": "mixed",
    "FIXED_RESET_PROBABILITY": "0.25",
    "UPRIGHT_RESET_PROBABILITY": "0.15",
    "FALLEN_VELOCITY_SCALE": "0.35",
    "TASK_PHASE": "recovery",
    "TARGET_FORWARD_VELOCITY": "0.8",
    "INIT_FROM_RUN_ID": "",
    "INIT_FROM_CHECKPOINT_STEP": "",
    "SAVE_REPLAY_BUFFER": "true",
    "MLFLOW_EXPERIMENT": "walker-ragdoll",
    "NUM_STEPS": "2048",
    "NUM_MINIBATCHES": "32",
    "UPDATE_EPOCHS": "10",
    "CLIP_COEF": "0.2",
    "ENT_COEF": "0.0",
    "VF_COEF": "0.5",
    "GAE_LAMBDA": "0.95",
    "MAX_GRAD_NORM": "0.5",
}


def get_env_or_default(key, default):
    return os.environ.get(key, default)


def parse_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args():
    parser = argparse.ArgumentParser(description="SAC / PPO Walker Ragdoll Training")
    parser.add_argument("--algo", type=str, default=get_env_or_default("ALGO", ENV_VARS["ALGO"]), choices=["sac", "ppo"], help="Training algorithm.")
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
    parser.add_argument(
        "--reset-mode",
        type=str,
        default=get_env_or_default("RESET_MODE", ENV_VARS["RESET_MODE"]),
        choices=["fixed", "mixed", "fallen", "upright"],
        help="Initial-state distribution for each episode.",
    )
    parser.add_argument(
        "--fixed-reset-probability",
        type=float,
        default=float(get_env_or_default("FIXED_RESET_PROBABILITY", ENV_VARS["FIXED_RESET_PROBABILITY"])),
        help="In mixed reset mode, fraction of episodes using the old fixed fallen pose.",
    )
    parser.add_argument(
        "--upright-reset-probability",
        type=float,
        default=float(get_env_or_default("UPRIGHT_RESET_PROBABILITY", ENV_VARS["UPRIGHT_RESET_PROBABILITY"])),
        help="In mixed reset mode, fraction of episodes starting almost upright.",
    )
    parser.add_argument(
        "--fallen-velocity-scale",
        type=float,
        default=float(get_env_or_default("FALLEN_VELOCITY_SCALE", ENV_VARS["FALLEN_VELOCITY_SCALE"])),
        help="Extra velocity noise applied to randomized fallen resets.",
    )
    parser.add_argument(
        "--task-phase",
        type=str,
        default=get_env_or_default("TASK_PHASE", ENV_VARS["TASK_PHASE"]),
        choices=["recovery", "balance", "walk", "target"],
        help="Reward curriculum phase: recovery, balance, walk, or target.",
    )
    parser.add_argument(
        "--target-forward-velocity",
        type=float,
        default=float(get_env_or_default("TARGET_FORWARD_VELOCITY", ENV_VARS["TARGET_FORWARD_VELOCITY"])),
        help="Target x velocity used by the walk phase.",
    )
    parser.add_argument(
        "--init-from-run-id",
        type=str,
        default=get_env_or_default("INIT_FROM_RUN_ID", ENV_VARS["INIT_FROM_RUN_ID"]),
        help="Initialize this new run from another SAC run's latest actor weights and obs normalization.",
    )
    parser.add_argument(
        "--init-from-checkpoint-step",
        type=int,
        default=int(get_env_or_default("INIT_FROM_CHECKPOINT_STEP", ENV_VARS["INIT_FROM_CHECKPOINT_STEP"]) or 0),
        help="Specific checkpoint step to use with --init-from-run-id. 0 means latest.",
    )
    parser.add_argument(
        "--actor-learning-starts",
        type=int,
        default=0,
        help="Step at which to start updating the actor. Useful for freezing a pre-trained actor while critics adapt.",
    )
    parser.add_argument(
        "--init-critics",
        action="store_true",
        default=True,
        help="Also initialize SAC critics from --init-from-run-id. Essential to prevent actor destruction.",
    )
    parser.set_defaults(
        save_replay_buffer=parse_bool(
            get_env_or_default("SAVE_REPLAY_BUFFER", ENV_VARS["SAVE_REPLAY_BUFFER"])
        )
    )
    parser.add_argument(
        "--save-replay-buffer",
        dest="save_replay_buffer",
        action="store_true",
        help="Save replay buffer inside checkpoints.",
    )
    parser.add_argument(
        "--no-save-replay-buffer",
        dest="save_replay_buffer",
        action="store_false",
        help="Do not save replay buffer inside checkpoints; useful for lighter target/walk checkpoints.",
    )
    parser.add_argument("--autotune", action="store_true", default=True)
    parser.add_argument("--no-autotune", dest="autotune", action="store_false")
    parser.add_argument("--capture-video", action="store_true", default=False)
    parser.add_argument("--mlflow-experiment", type=str, default=get_env_or_default("MLFLOW_EXPERIMENT", ENV_VARS["MLFLOW_EXPERIMENT"]))
    parser.add_argument("--disable-mlflow", action="store_true", default=False)
    parser.add_argument("--allow-mismatched-env-version", action="store_true", default=False)
    # PPO-specific arguments
    parser.add_argument("--num-steps", type=int, default=int(get_env_or_default("NUM_STEPS", ENV_VARS["NUM_STEPS"])), help="PPO rollout length per env.")
    parser.add_argument("--num-minibatches", type=int, default=int(get_env_or_default("NUM_MINIBATCHES", ENV_VARS["NUM_MINIBATCHES"])), help="PPO number of minibatches.")
    parser.add_argument("--update-epochs", type=int, default=int(get_env_or_default("UPDATE_EPOCHS", ENV_VARS["UPDATE_EPOCHS"])), help="PPO update epochs per rollout.")
    parser.add_argument("--clip-coef", type=float, default=float(get_env_or_default("CLIP_COEF", ENV_VARS["CLIP_COEF"])), help="PPO clip coefficient.")
    parser.add_argument("--ent-coef", type=float, default=float(get_env_or_default("ENT_COEF", ENV_VARS["ENT_COEF"])), help="PPO entropy coefficient.")
    parser.add_argument("--vf-coef", type=float, default=float(get_env_or_default("VF_COEF", ENV_VARS["VF_COEF"])), help="PPO value function coefficient.")
    parser.add_argument("--gae-lambda", type=float, default=float(get_env_or_default("GAE_LAMBDA", ENV_VARS["GAE_LAMBDA"])), help="PPO GAE lambda.")
    parser.add_argument("--max-grad-norm", type=float, default=float(get_env_or_default("MAX_GRAD_NORM", ENV_VARS["MAX_GRAD_NORM"])), help="PPO max gradient norm.")
    parser.add_argument("--norm-adv", action="store_true", default=True, help="PPO normalize advantages.")
    parser.add_argument("--no-norm-adv", dest="norm_adv", action="store_false")
    return parser.parse_args()


def make_env(
    env_id,
    idx,
    capture_video,
    run_name,
    reset_mode="mixed",
    fixed_reset_probability=0.25,
    upright_reset_probability=0.15,
    fallen_velocity_scale=0.35,
    task_phase="recovery",
    target_forward_velocity=0.8,
):
    def thunk():
        env_kwargs = {
            "reset_mode": reset_mode,
            "fixed_reset_probability": fixed_reset_probability,
            "upright_reset_probability": upright_reset_probability,
            "fallen_velocity_scale": fallen_velocity_scale,
            "task_phase": task_phase,
            "target_forward_velocity": target_forward_velocity,
        }
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, **env_kwargs)
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


def get_normalize_observation_wrapper(env):
    current = env
    while current is not None:
        if isinstance(current, gym.wrappers.NormalizeObservation):
            return current
        current = getattr(current, "env", None)
    raise RuntimeError("NormalizeObservation wrapper not found in env stack.")


def get_obs_rms(envs):
    return get_normalize_observation_wrapper(envs).obs_rms


def set_obs_rms(envs, obs_rms):
    get_normalize_observation_wrapper(envs).obs_rms = obs_rms


def adapt_obs_rms(obs_rms, target_shape):
    if obs_rms is None or tuple(obs_rms.mean.shape) == tuple(target_shape):
        return obs_rms

    adapted = copy.deepcopy(obs_rms)
    old_mean = np.asarray(obs_rms.mean)
    old_var = np.asarray(obs_rms.var)
    new_mean = np.zeros(target_shape, dtype=old_mean.dtype)
    new_var = np.ones(target_shape, dtype=old_var.dtype)
    copy_len = min(old_mean.size, new_mean.size)
    new_mean.reshape(-1)[:copy_len] = old_mean.reshape(-1)[:copy_len]
    new_var.reshape(-1)[:copy_len] = old_var.reshape(-1)[:copy_len]
    adapted.mean = new_mean
    adapted.var = new_var
    return adapted


def load_state_dict_with_expanded_input(module, source_state_dict, input_weight_key):
    current_state_dict = module.state_dict()
    copied = []
    expanded = False

    for key, source_value in source_state_dict.items():
        if key not in current_state_dict:
            continue
        target_value = current_state_dict[key]
        if source_value.shape == target_value.shape:
            current_state_dict[key] = source_value
            copied.append(key)
        elif key == input_weight_key and source_value.ndim == 2 and target_value.ndim == 2:
            copy_rows = min(source_value.shape[0], target_value.shape[0])
            copy_cols = min(source_value.shape[1], target_value.shape[1])
            # Zero out the expanded portion so new inputs don't inject random noise
            target_value.data.zero_()
            target_value.data[:copy_rows, :copy_cols] = source_value[:copy_rows, :copy_cols]
            current_state_dict[key] = target_value
            copied.append(key)
            expanded = True

    module.load_state_dict(current_state_dict)
    return copied, expanded


def load_actor_initialization(actor, checkpoint):
    copied, expanded = load_state_dict_with_expanded_input(
        actor, checkpoint["actor_state_dict"], "backbone.0.weight"
    )
    if "backbone.0.weight" not in copied:
        raise ValueError("Could not initialize actor first layer from checkpoint.")
    return expanded


def reset_envs_without_obs_rms_update(envs, seed):
    """Reset the vector env while preserving restored obs_rms on the first resume step."""
    normalize_env = get_normalize_observation_wrapper(envs)
    raw_env = normalize_env.env
    raw_obs, info = raw_env.reset(seed=seed)
    normalized_obs = (raw_obs - normalize_env.obs_rms.mean) / np.sqrt(
        normalize_env.obs_rms.var + normalize_env.epsilon
    )
    return np.clip(normalized_obs, -10, 10), info


def start_mlflow_run(args, run_name):
    if args.disable_mlflow or mlflow is None:
        if mlflow is None and not args.disable_mlflow:
            print("[MLFLOW] warning: mlflow is not installed; tracking disabled.")
        return False
    try:
        mlflow.set_experiment(args.mlflow_experiment)
        mlflow.start_run(run_name=run_name)
        mlflow.log_params(vars(args))
        print(f"[MLFLOW] tracking started in experiment: {args.mlflow_experiment}")
        return True
    except Exception as exc:
        print(f"[MLFLOW] warning: could not start tracking: {exc}")
        return False


def log_mlflow_metrics(metrics, step, enabled):
    if not enabled:
        return
    try:
        mlflow.log_metrics(metrics, step=step)
    except Exception as exc:
        print(f"[MLFLOW] warning: could not log metrics: {exc}")


def log_mlflow_artifact(path, enabled):
    if not enabled:
        return
    try:
        mlflow.log_artifact(path)
    except Exception as exc:
        print(f"[MLFLOW] warning: could not log artifact {path}: {exc}")


def end_mlflow_run(enabled):
    if enabled:
        mlflow.end_run()


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
    task_phase=None,
    target_forward_velocity=None,
    save_replay_buffer=True,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    replay_buffer_state = (
        serialize_replay_buffer(replay_buffer) if save_replay_buffer else None
    )
    checkpoint = {
            "algo": "sac",
            "env_version": ENV_VERSION,
            "task_phase": task_phase,
            "target_forward_velocity": target_forward_velocity,
            "global_step": global_step,
            "num_envs": envs.num_envs,
            "buffer_size": replay_buffer.size,
            "replay_buffer_saved": replay_buffer_state is not None,
            "actor_state_dict": actor.state_dict(),
            "qf1_state_dict": qf1.state_dict(),
            "qf2_state_dict": qf2.state_dict(),
            "qf1_target_state_dict": qf1_target.state_dict(),
            "qf2_target_state_dict": qf2_target.state_dict(),
            "actor_optimizer_state_dict": actor_optimizer.state_dict(),
            "q_optimizer_state_dict": q_optimizer.state_dict(),
            "alpha_optimizer_state_dict": alpha_optimizer.state_dict() if alpha_optimizer is not None else None,
            "log_alpha": log_alpha.detach().cpu(),
            "obs_rms": get_obs_rms(envs),
            "replay_buffer": replay_buffer_state,
            "rng_state": get_rng_state(),
        }
    try:
        torch.save(checkpoint, path)
    except MemoryError:
        if not replay_buffer_state:
            raise
        print(
            "[CHECKPOINT] MemoryError while saving replay buffer; retrying with a lightweight checkpoint."
        )
        checkpoint["replay_buffer"] = None
        checkpoint["replay_buffer_saved"] = False
        replay_buffer_state = None
        gc.collect()
        if os.path.exists(path):
            os.remove(path)
        torch.save(checkpoint, path)
    suffix = "" if checkpoint["replay_buffer_saved"] else " (without replay buffer)"
    print(f"[CHECKPOINT] Saved at step {global_step} -> {path}{suffix}")

    actor_path = os.path.join(os.path.dirname(path), f"sac_actor_{global_step}.pt")
    torch.save(
        {
            "algo": "sac_actor",
            "env_version": ENV_VERSION,
            "task_phase": task_phase,
            "target_forward_velocity": target_forward_velocity,
            "global_step": global_step,
            "actor_state_dict": actor.state_dict(),
            "obs_rms": get_obs_rms(envs),
        },
        actor_path,
    )
    print(f"[CHECKPOINT] Saved lightweight actor -> {actor_path}")


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


def latest_sac_actor_checkpoint(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        return None
    candidates = []
    for name in os.listdir(ckpt_dir):
        if name.startswith("sac_actor_") and name.endswith(".pt"):
            try:
                candidates.append((int(name[len("sac_actor_") : -3]), os.path.join(ckpt_dir, name)))
            except ValueError:
                pass
    return max(candidates)[1] if candidates else None


def load_torch_checkpoint(path, device):
    return torch.load(path, map_location=device)


def resolve_sac_checkpoint(run_id, checkpoint_step=0, prefer_actor=False):
    ckpt_dir = get_checkpoint_dir(run_id)
    if checkpoint_step:
        if prefer_actor:
            actor_path = os.path.join(ckpt_dir, f"sac_actor_{checkpoint_step}.pt")
            if os.path.exists(actor_path):
                return actor_path
        ckpt_path = os.path.join(ckpt_dir, f"sac_ckpt_{checkpoint_step}.pt")
        if os.path.exists(ckpt_path):
            return ckpt_path
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if prefer_actor:
        actor_path = latest_sac_actor_checkpoint(ckpt_dir)
        if actor_path is not None:
            return actor_path
    return latest_sac_checkpoint(ckpt_dir)


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

def ppo_layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOAgent(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.critic = nn.Sequential(
            ppo_layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            ppo_layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            ppo_layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            ppo_layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            ppo_layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            ppo_layer_init(nn.Linear(256, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, obs):
        return self.critic(obs)

    def get_action_and_value(self, obs, action=None):
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = torch.distributions.Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)

    def get_deterministic_action(self, obs):
        return self.actor_mean(obs)


def save_ppo_checkpoint(path, global_step, agent, optimizer, envs, task_phase=None, target_forward_velocity=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "algo": "ppo",
        "env_version": ENV_VERSION,
        "task_phase": task_phase,
        "target_forward_velocity": target_forward_velocity,
        "global_step": global_step,
        "agent_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "obs_rms": get_obs_rms(envs),
        "rng_state": get_rng_state(),
    }
    torch.save(checkpoint, path)
    print(f"[CHECKPOINT] Saved PPO at step {global_step} -> {path}")


def latest_ppo_checkpoint(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        return None
    candidates = []
    for name in os.listdir(ckpt_dir):
        if name.startswith("ppo_ckpt_") and name.endswith(".pt"):
            try:
                candidates.append((int(name[len("ppo_ckpt_"):-3]), os.path.join(ckpt_dir, name)))
            except ValueError:
                pass
    return max(candidates)[1] if candidates else None


def latest_checkpoint_any(ckpt_dir):
    """Find latest checkpoint of any algorithm type."""
    sac = latest_sac_checkpoint(ckpt_dir)
    ppo = latest_ppo_checkpoint(ckpt_dir)
    if sac is None:
        return ppo
    if ppo is None:
        return sac
    sac_step = int(os.path.basename(sac).split("_")[-1].replace(".pt", ""))
    ppo_step = int(os.path.basename(ppo).split("_")[-1].replace(".pt", ""))
    return sac if sac_step >= ppo_step else ppo


def train_ppo(start_time=None):
    if start_time is None:
        start_time = time.time()
    args = parse_args()
    run_name = f"{args.run_id}__{args.seed}"

    if not args.resume or args.force:
        force_delete_run(args.run_id)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env_id = "WalkerRagdoll-v0"
    envs = gym.vector.SyncVectorEnv([
        make_env(env_id, i, args.capture_video, run_name,
                 reset_mode=args.reset_mode,
                 fixed_reset_probability=args.fixed_reset_probability,
                 upright_reset_probability=args.upright_reset_probability,
                 fallen_velocity_scale=args.fallen_velocity_scale,
                 task_phase=args.task_phase,
                 target_forward_velocity=args.target_forward_velocity)
        for i in range(args.num_envs)
    ])
    envs = gym.wrappers.NormalizeObservation(envs)
    envs = gym.wrappers.TransformObservation(envs, lambda obs: np.clip(obs, -10, 10))
    envs = gym.wrappers.NormalizeReward(envs, gamma=args.gamma)
    envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))

    obs_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(np.prod(envs.single_action_space.shape))
    agent = PPOAgent(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    global_step = 0
    restored_obs_rms = None

    ckpt_dir = get_checkpoint_dir(args.run_id)
    if args.resume:
        ckpt_path = latest_ppo_checkpoint(ckpt_dir)
        if ckpt_path:
            print(f"[CHECKPOINT] Loading PPO from {ckpt_path}")
            checkpoint = load_torch_checkpoint(ckpt_path, device)
            if checkpoint.get("algo") != "ppo":
                raise ValueError(f"Checkpoint {ckpt_path} is not a PPO checkpoint.")
            agent.load_state_dict(checkpoint["agent_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if checkpoint.get("obs_rms") is not None:
                restored_obs_rms = adapt_obs_rms(checkpoint["obs_rms"], envs.single_observation_space.shape)
                set_obs_rms(envs, restored_obs_rms)
            if checkpoint.get("rng_state") is not None:
                set_rng_state(checkpoint["rng_state"])
            global_step = int(checkpoint.get("global_step", 0))

    run_dir = get_run_dir(run_name)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(run_dir, purge_step=global_step if args.resume else None)
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()), global_step)
    mlflow_enabled = start_mlflow_run(args, run_name)

    batch_size = int(args.num_envs * args.num_steps)
    minibatch_size = int(batch_size // args.num_minibatches)
    num_updates = args.total_timesteps // batch_size

    # Rollout storage
    obs_buf = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions_buf = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)

    if restored_obs_rms is not None:
        next_obs, _ = reset_envs_without_obs_rms_update(envs, seed=args.seed)
    else:
        next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs).to(device)
    start_update = global_step // batch_size
    next_checkpoint_step = ((global_step // args.checkpoint_interval) + 1) * args.checkpoint_interval

    try:
        for update in range(start_update, num_updates):
            # Anneal LR
            frac = 1.0 - update / num_updates
            lr_now = frac * args.learning_rate
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            for step in range(args.num_steps):
                global_step += args.num_envs
                obs_buf[step] = next_obs
                dones_buf[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values_buf[step] = value.flatten()
                actions_buf[step] = action
                logprobs_buf[step] = logprob

                next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                rewards_buf[step] = torch.as_tensor(reward, dtype=torch.float32, device=device)
                next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
                next_done = torch.as_tensor(np.logical_or(terminations, truncations), dtype=torch.float32, device=device)

                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            ep_r = float(info["episode"]["r"])
                            ep_l = float(info["episode"]["l"])
                            print(f"global_step={global_step}, episodic_return={ep_r:.2f}, episodic_length={ep_l:.0f}")
                            writer.add_scalar("charts/episodic_return", ep_r, global_step)
                            writer.add_scalar("charts/episodic_length", ep_l, global_step)
                            log_mlflow_metrics({"episodic_return": ep_r, "episodic_length": ep_l}, global_step, mlflow_enabled)

            # GAE
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards_buf).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones_buf[t + 1]
                        nextvalues = values_buf[t + 1]
                    delta = rewards_buf[t] + args.gamma * nextvalues * nextnonterminal - values_buf[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values_buf

            # Flatten
            b_obs = obs_buf.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs_buf.reshape(-1)
            b_actions = actions_buf.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values_buf.reshape(-1)

            # Optimize
            b_inds = np.arange(batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

            # Logging
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            writer.add_scalar("charts/learning_rate", lr_now, global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            log_mlflow_metrics({
                "value_loss": v_loss.item(), "policy_loss": pg_loss.item(),
                "entropy": entropy_loss.item(), "clipfrac": np.mean(clipfracs),
                "explained_variance": explained_var,
                "sps": int(global_step / (time.time() - start_time)),
            }, global_step, mlflow_enabled)

            if global_step >= next_checkpoint_step:
                save_ppo_checkpoint(
                    os.path.join(ckpt_dir, f"ppo_ckpt_{global_step}.pt"),
                    global_step, agent, optimizer, envs,
                    task_phase=args.task_phase, target_forward_velocity=args.target_forward_velocity,
                )
                log_mlflow_artifact(os.path.join(ckpt_dir, f"ppo_ckpt_{global_step}.pt"), mlflow_enabled)
                next_checkpoint_step += args.checkpoint_interval

    except KeyboardInterrupt:
        print("\n[TRAIN] Interrupted by user. Saving checkpoint...")
    finally:
        final_path = os.path.join(ckpt_dir, f"ppo_ckpt_{global_step}.pt")
        save_ppo_checkpoint(final_path, global_step, agent, optimizer, envs,
                            task_phase=args.task_phase, target_forward_velocity=args.target_forward_velocity)
        log_mlflow_artifact(final_path, mlflow_enabled)
        envs.close()
        writer.close()
        end_mlflow_run(mlflow_enabled)
        print(f"PPO training completed. Total steps: {global_step}")


def train(start_time=None):
    if start_time is None:
        start_time = time.time()
    args = parse_args()
    run_name = f"{args.run_id}__{args.seed}"

    if not args.resume or args.force:
        force_delete_run(args.run_id)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env_id = "WalkerRagdoll-v0"
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                env_id,
                i,
                args.capture_video,
                run_name,
                reset_mode=args.reset_mode,
                fixed_reset_probability=args.fixed_reset_probability,
                upright_reset_probability=args.upright_reset_probability,
                fallen_velocity_scale=args.fallen_velocity_scale,
                task_phase=args.task_phase,
                target_forward_velocity=args.target_forward_velocity,
            )
            for i in range(args.num_envs)
        ]
    )
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
            checkpoint = load_torch_checkpoint(ckpt_path, device)
            if checkpoint.get("algo") != "sac":
                raise ValueError(f"Checkpoint {ckpt_path} is not a SAC checkpoint.")
            checkpoint_env_version = checkpoint.get("env_version")
            if checkpoint_env_version != ENV_VERSION and not args.allow_mismatched_env_version:
                raise ValueError(
                    "Checkpoint environment version mismatch: "
                    f"checkpoint={checkpoint_env_version!r} current={ENV_VERSION!r}. "
                    "Start a new run with --force, or pass --allow-mismatched-env-version "
                    "only if you intentionally want to fine-tune across a reward change."
                )
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
                restored_obs_rms = adapt_obs_rms(
                    checkpoint["obs_rms"], envs.single_observation_space.shape
                )
                set_obs_rms(envs, restored_obs_rms)
            if checkpoint.get("rng_state") is not None:
                set_rng_state(checkpoint["rng_state"])
            global_step = int(checkpoint.get("global_step", 0))
    elif args.init_from_run_id:
        ckpt_path = resolve_sac_checkpoint(
            args.init_from_run_id,
            args.init_from_checkpoint_step,
            prefer_actor=not args.init_critics,
        )
        if ckpt_path is None:
            raise FileNotFoundError(
                f"No SAC checkpoint found for init-from run '{args.init_from_run_id}'."
            )
        print(f"[INIT] Loading actor initialization from {ckpt_path}")
        checkpoint = load_torch_checkpoint(ckpt_path, device)
        if checkpoint.get("algo") not in {"sac", "sac_actor"}:
            raise ValueError(f"Checkpoint {ckpt_path} is not a SAC checkpoint.")
        actor_expanded = load_actor_initialization(actor, checkpoint)
        if actor_expanded:
            print("[INIT] Expanded actor input layer for new target observations.")
        if args.init_critics and checkpoint.get("algo") == "sac":
            print("[INIT] Loading critics too; expanding if observation size changed.")
            load_state_dict_with_expanded_input(qf1, checkpoint["qf1_state_dict"], "net.0.weight")
            load_state_dict_with_expanded_input(qf2, checkpoint["qf2_state_dict"], "net.0.weight")
            load_state_dict_with_expanded_input(qf1_target, checkpoint["qf1_target_state_dict"], "net.0.weight")
            load_state_dict_with_expanded_input(qf2_target, checkpoint["qf2_target_state_dict"], "net.0.weight")
        elif args.init_critics:
            print("[INIT] Skipping critic initialization because lightweight actor checkpoint was loaded.")
        if checkpoint.get("obs_rms") is not None:
            restored_obs_rms = adapt_obs_rms(
                checkpoint["obs_rms"], envs.single_observation_space.shape
            )
            set_obs_rms(envs, restored_obs_rms)
            print("[INIT] Loaded observation normalization statistics.")
        print(
            "[INIT] Starting a fresh replay buffer and optimizer state for the new reward phase."
        )

    run_dir = get_run_dir(run_name)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(run_dir, purge_step=global_step if args.resume else None)
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()), global_step)
    mlflow_enabled = start_mlflow_run(args, run_name)

    if restored_obs_rms is not None:
        next_obs, _ = reset_envs_without_obs_rms_update(envs, seed=args.seed)
    else:
        next_obs, _ = envs.reset(seed=args.seed)
    next_obs = next_obs.astype(np.float32)
    next_checkpoint_step = ((global_step // args.checkpoint_interval) + 1) * args.checkpoint_interval

    try:
        while global_step < args.total_timesteps:
            obs = next_obs
            if global_step < args.learning_starts and not args.init_from_run_id:
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
                        log_mlflow_metrics(
                            {
                                "episodic_return": ep_r,
                                "episodic_length": ep_l,
                            },
                            global_step,
                            mlflow_enabled,
                        )

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
                nn.utils.clip_grad_norm_(list(qf1.parameters()) + list(qf2.parameters()), 1.0)
                q_optimizer.step()

                if global_step % (args.policy_frequency * args.num_envs) == 0 and global_step > args.actor_learning_starts:
                    for _ in range(args.policy_frequency):
                        pi, log_pi, _ = actor.get_action(batch.obs)
                        qf1_pi = qf1(batch.obs, pi)
                        qf2_pi = qf2(batch.obs, pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)
                        actor_loss = ((log_alpha.exp() * log_pi) - min_qf_pi).mean()

                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
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
                    log_mlflow_metrics(
                        {
                            "qf1_values": qf1_a_values.mean().item(),
                            "qf2_values": qf2_a_values.mean().item(),
                            "qf_loss": qf_loss.item(),
                            "alpha": log_alpha.exp().item(),
                            "sps": int(global_step / (time.time() - start_time)),
                        },
                        global_step,
                        mlflow_enabled,
                    )

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
                    task_phase=args.task_phase,
                    target_forward_velocity=args.target_forward_velocity,
                    save_replay_buffer=args.save_replay_buffer,
                )
                log_mlflow_artifact(
                    os.path.join(ckpt_dir, f"sac_ckpt_{global_step}.pt"),
                    mlflow_enabled,
                )
                next_checkpoint_step += args.checkpoint_interval
    except KeyboardInterrupt:
        print("\n[TRAIN] Interrupted by user. Saving checkpoint...")
    finally:
        final_checkpoint_path = os.path.join(ckpt_dir, f"sac_ckpt_{global_step}.pt")
        save_sac_checkpoint(
            final_checkpoint_path,
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
            task_phase=args.task_phase,
            target_forward_velocity=args.target_forward_velocity,
            save_replay_buffer=args.save_replay_buffer,
        )
        log_mlflow_artifact(final_checkpoint_path, mlflow_enabled)
        if mlflow_enabled:
            try:
                mlflow.pytorch.log_model(actor, "actor_final")
            except Exception as exc:
                print(f"[MLFLOW] warning: could not log final actor model: {exc}")
        envs.close()
        writer.close()
        end_mlflow_run(mlflow_enabled)
        print(f"Training completed. Total steps: {global_step}")



if __name__ == "__main__":
    import sys
    # Quick peek at --algo without full parse to dispatch correctly
    algo = "sac"
    for i, arg in enumerate(sys.argv):
        if arg == "--algo" and i + 1 < len(sys.argv):
            algo = sys.argv[i + 1]
    if algo == "ppo":
        train_ppo(start_time=time.time())
    else:
        train(start_time=time.time())
