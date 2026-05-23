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
    force_delete_run,
    get_checkpoint_dir,
    get_rng_state,
    set_rng_state,
    ENV_VERSION,
    start_mlflow_run,
    log_mlflow_metrics,
    log_mlflow_artifact,
    end_mlflow_run,
)

# Symlog scaling functions used in DreamerV3 to stabilize targets
def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

def symexp(y):
    return torch.sign(y) * (torch.exp(torch.abs(y)) - 1.0)

class RSSM(nn.Module):
    """Recurrent State-Space Model (RSSM) representing deterministic & stochastic dynamics."""
    def __init__(self, action_dim, hidden_dim=256, stochastic_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.stochastic_dim = stochastic_dim
        
        # Deterministic state GRU cell: h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})
        self.gru_cell = nn.GRUCell(stochastic_dim + action_dim, hidden_dim)
        
        # Prior network: p(z_t | h_t)
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ELU(),
            nn.Linear(128, stochastic_dim * 2) # mean and logstd
        )
        
        # Posterior network: q(z_t | h_t, e_t)
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + 128, 128),
            nn.ELU(),
            nn.Linear(128, stochastic_dim * 2) # mean and logstd
        )

    def initial_state(self, batch_size, device):
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        z = torch.zeros(batch_size, self.stochastic_dim, device=device)
        return h, z

    def transition(self, h, z, action):
        # Action shape: (B, action_dim)
        # Stochastic state shape: (B, stochastic_dim)
        input_gru = torch.cat([z, action], dim=-1)
        h_next = self.gru_cell(input_gru, h)
        
        # Compute prior distribution
        prior_out = self.prior_net(h_next)
        prior_mean, prior_logstd = torch.chunk(prior_out, 2, dim=-1)
        prior_std = F.softplus(prior_logstd) + 0.1
        
        # Sample stochastic state
        eps = torch.randn_like(prior_mean)
        z_next = prior_mean + eps * prior_std
        return h_next, z_next, prior_mean, prior_std

    def posterior(self, h_next, embed):
        # embed: observation features (B, 128)
        post_input = torch.cat([h_next, embed], dim=-1)
        post_out = self.posterior_net(post_input)
        post_mean, post_logstd = torch.chunk(post_out, 2, dim=-1)
        post_std = F.softplus(post_logstd) + 0.1
        
        # Sample stochastic state
        eps = torch.randn_like(post_mean)
        z_next = post_mean + eps * post_std
        return z_next, post_mean, post_std

class WorldModel(nn.Module):
    """World Model encapsulating Encoder, Decoder, RSSM, and reward/continue predictors."""
    def __init__(self, obs_dim, action_dim, hidden_dim=256, stochastic_dim=32):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU()
        )
        
        # RSSM
        self.rssm = RSSM(action_dim, hidden_dim, stochastic_dim)
        
        # Observation Decoder (reconstructs symlog observation)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + stochastic_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, obs_dim)
        )
        
        # Reward Predictor
        self.reward_net = nn.Sequential(
            nn.Linear(hidden_dim + stochastic_dim, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        )
        
        # Continue Predictor (termination probability tracker)
        self.continue_net = nn.Sequential(
            nn.Linear(hidden_dim + stochastic_dim, 128),
            nn.ELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, obs_seq, action_seq, done_seq):
        # Batch training
        # obs_seq: (L, B, obs_dim)
        # action_seq: (L-1, B, action_dim)
        # done_seq: (L-1, B, 1)
        L, B, _ = obs_seq.shape
        device = obs_seq.device
        
        embeds = self.encoder(obs_seq) # (L, B, 128)
        
        h, z = self.rssm.initial_state(B, device)
        
        h_list, z_list = [h], [z]
        prior_means, prior_stds = [], []
        post_means, post_stds = [], []
        
        # Initial step posterior
        z, post_mean, post_std = self.rssm.posterior(h, embeds[0])
        z_list[0] = z
        post_means.append(post_mean)
        post_stds.append(post_std)
        
        # Scan through the sequence
        for t in range(L - 1):
            h, _, p_m, p_s = self.rssm.transition(h, z, action_seq[t])
            # If the episode terminated, we reset deterministic state
            reset_mask = done_seq[t].float()
            h = h * (1.0 - reset_mask)
            
            z, q_m, q_s = self.rssm.posterior(h, embeds[t+1])
            
            h_list.append(h)
            z_list.append(z)
            prior_means.append(p_m)
            prior_stds.append(p_s)
            post_means.append(q_m)
            post_stds.append(q_s)
            
        h_stack = torch.stack(h_list, dim=0) # (L, B, hidden_dim)
        z_stack = torch.stack(z_list, dim=0) # (L, B, stochastic_dim)
        
        return h_stack, z_stack, prior_means, prior_stds, post_means, post_stds

class DreamerActor(nn.Module):
    """Continuous Actor mapping latent states (h, z) to actions."""
    def __init__(self, hidden_dim, stochastic_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + stochastic_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, action_dim * 2) # mean and logstd
        )

    def forward(self, h, z):
        state = torch.cat([h, z], dim=-1)
        out = self.net(state)
        mean, logstd = torch.chunk(out, 2, dim=-1)
        logstd = torch.clamp(logstd, -20, 2)
        std = torch.exp(logstd)
        return mean, std

    def get_action(self, h, z, sample=True):
        mean, std = self.forward(h, z)
        if sample:
            eps = torch.randn_like(mean)
            action = mean + eps * std
        else:
            action = mean
        # Return action squashed with tanh
        return torch.tanh(action)

class DreamerCritic(nn.Module):
    """Critic network mapping latent states (h, z) to value estimates."""
    def __init__(self, hidden_dim, stochastic_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + stochastic_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        )

    def forward(self, h, z):
        state = torch.cat([h, z], dim=-1)
        return self.net(state)

class SequenceReplayBuffer:
    """Replay buffer that stores sequences of transitions."""
    def __init__(self, capacity, obs_shape, action_shape, device):
        self.capacity = capacity
        self.device = device
        
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.bool_)
        
        self.idx = 0
        self.size = 0

    def add(self, obs, action, reward, done):
        self.obs[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, seq_len):
        # We need to sample sequences of length `seq_len`
        obs_batch, action_batch, reward_batch, done_batch = [], [], [], []
        
        for _ in range(batch_size):
            # Pick a random starting index
            valid = False
            while not valid:
                start_idx = np.random.randint(0, self.size - seq_len)
                # Check that the sequence does not wrap around the buffer boundary
                if (start_idx + seq_len) <= self.size and start_idx < self.idx <= (start_idx + seq_len):
                    continue
                valid = True
                
            obs_batch.append(self.obs[start_idx : start_idx + seq_len])
            action_batch.append(self.actions[start_idx : start_idx + seq_len - 1])
            reward_batch.append(self.rewards[start_idx : start_idx + seq_len - 1])
            done_batch.append(self.dones[start_idx : start_idx + seq_len - 1])
            
        # Transpose batches to match sequence-first shape (L, B, dim)
        obs_t = torch.as_tensor(np.array(obs_batch), device=self.device).permute(1, 0, 2)
        action_t = torch.as_tensor(np.array(action_batch), device=self.device).permute(1, 0, 2)
        reward_t = torch.as_tensor(np.array(reward_batch), device=self.device)
        done_t = torch.as_tensor(np.array(done_batch), device=self.device)
        
        return obs_t, action_t, reward_t.permute(1, 0, 2), done_t.permute(1, 0, 2)

def parse_dreamer_args():
    parser = argparse.ArgumentParser(description="DreamerV3 Walker Ragdoll Training")
    parser.add_argument("--run-id", type=str, default="walker_dreamer_1m")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--total-timesteps", type=int, default=1000000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-envs", type=int, default=4) # Smaller number of envs since sequential data logging is seq-based
    parser.add_argument("--buffer-size", type=int, default=200000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--checkpoint-interval", type=int, default=200000)
    parser.add_argument("--reset-mode", type=str, default="upright")
    parser.add_argument("--fixed-reset-probability", type=float, default=0.25)
    parser.add_argument("--upright-reset-probability", type=float, default=0.15)
    parser.add_argument("--fallen-velocity-scale", type=float, default=0.35)
    parser.add_argument("--task-phase", type=str, default="target")
    parser.add_argument("--target-forward-velocity", type=float, default=0.8)
    
    # DreamerV3 specific hyperparameters
    parser.add_argument("--seq-len", type=int, default=50, help="World model sequence batch training length (L)")
    parser.add_argument("--imag-horizon", type=int, default=15, help="Imagination sequence length (H)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size of sequences")
    parser.add_argument("--kl-weight", type=float, default=1.0, help="KL divergence regularization weight")
    
    return parser.parse_args()

def save_dreamer_checkpoint(
    ckpt_path,
    global_step,
    model,
    actor,
    critic,
    model_opt,
    actor_opt,
    critic_opt,
    envs,
    task_phase,
    target_forward_velocity,
):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    # Save actor only
    actor_only_path = ckpt_path.replace("dreamer_ckpt_", "dreamer_actor_")
    
    torch.save({
        "algo": "dreamer",
        "actor_state_dict": actor.state_dict(),
        "rssm_state_dict": model.rssm.state_dict(),
        "encoder_state_dict": model.encoder.state_dict(),
        "obs_rms": envs.obs_rms if hasattr(envs, "obs_rms") else None,
    }, actor_only_path)
    
    # Save full state
    state = {
        "algo": "dreamer",
        "env_version": ENV_VERSION,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "model_opt_state_dict": model_opt.state_dict(),
        "actor_opt_state_dict": actor_opt.state_dict(),
        "critic_opt_state_dict": critic_opt.state_dict(),
        "obs_rms": envs.obs_rms if hasattr(envs, "obs_rms") else None,
        "task_phase": task_phase,
        "target_forward_velocity": target_forward_velocity,
        "rng_state": get_rng_state(),
    }
    torch.save(state, ckpt_path)
    print(f"[CHECKPOINT] Saved at step {global_step} -> {ckpt_path}")

def train_dreamer():
    args = parse_dreamer_args()
    run_name = f"{args.run_id}__{args.seed}"
    start_time = time.time()

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
                False,
                run_name,
                reset_mode=args.reset_mode,
                fixed_reset_probability=args.fixed_reset_probability,
                upright_reset_probability=args.upright_reset_probability,
                fallen_velocity_scale=args.fallen_velocity_scale,
                task_phase=args.task_phase,
                target_forward_velocity=args.target_forward_velocity,
                terminate_when_unhealthy=(args.task_phase == "target"),
            )
            for i in range(args.num_envs)
        ]
    )
    envs = gym.wrappers.NormalizeObservation(envs)
    envs = gym.wrappers.TransformObservation(envs, lambda obs: np.clip(obs, -10, 10))

    obs_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(np.prod(envs.single_action_space.shape))

    # Define Dreamer architectures
    model = WorldModel(obs_dim, action_dim).to(device)
    actor = DreamerActor(hidden_dim=256, stochastic_dim=32, action_dim=action_dim).to(device)
    critic = DreamerCritic(hidden_dim=256, stochastic_dim=32).to(device)

    # Optimizers
    model_opt = optim.Adam(model.parameters(), lr=args.learning_rate)
    actor_opt = optim.Adam(actor.parameters(), lr=args.learning_rate)
    critic_opt = optim.Adam(critic.parameters(), lr=args.learning_rate)

    rb = SequenceReplayBuffer(args.buffer_size, envs.single_observation_space.shape, envs.single_action_space.shape, device)
    global_step = 0
    restored_obs_rms = None
    ckpt_dir = get_checkpoint_dir(args.run_id)

    # Setup WandB
            mode="offline",
            dir=".",
        )

    # Setup MLflow
    mlf_run = start_mlflow_run(args, run_name, "dreamer")

    # Resume handling
    if args.resume:
        files = [f for f in os.listdir(ckpt_dir) if f.startswith("dreamer_ckpt_") and f.endswith(".pt")] if os.path.exists(ckpt_dir) else []
        if files:
            files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            ckpt_path = os.path.join(ckpt_dir, files[-1])
            print(f"[CHECKPOINT] Loading from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            actor.load_state_dict(checkpoint["actor_state_dict"])
            critic.load_state_dict(checkpoint["critic_state_dict"])
            model_opt.load_state_dict(checkpoint["model_opt_state_dict"])
            actor_opt.load_state_dict(checkpoint["actor_opt_state_dict"])
            critic_opt.load_state_dict(checkpoint["critic_opt_state_dict"])
            if checkpoint.get("obs_rms") is not None:
                from train_walker import set_obs_rms, adapt_obs_rms
                restored_obs_rms = adapt_obs_rms(checkpoint["obs_rms"], envs.single_observation_space.shape)
                set_obs_rms(envs, restored_obs_rms)
            if checkpoint.get("rng_state") is not None:
                set_rng_state(checkpoint["rng_state"])
            global_step = checkpoint["global_step"]

    writer = SummaryWriter(os.path.join("runs", run_name))

    obs = envs.reset()[0]
    
    # Recurrent state trackers for active simulation environment lanes
    h_eval, z_eval = model.rssm.initial_state(args.num_envs, device)
    
    next_checkpoint_step = ((global_step // args.checkpoint_interval) + 1) * args.checkpoint_interval

    try:
        while global_step < args.total_timesteps:
            # 1. Action selection using the current actor and RSSM
            if global_step < 5000:
                actions = np.array([envs.single_action_space.sample() for _ in range(args.num_envs)])
            else:
                with torch.no_grad():
                    actions = actor.get_action(h_eval, z_eval, sample=True).cpu().numpy()

            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            
            # Step the RSSM recurrent states forward
            with torch.no_grad():
                actions_t = torch.as_tensor(actions, dtype=torch.float32, device=device)
                embeds = model.encoder(torch.as_tensor(next_obs, dtype=torch.float32, device=device))
                h_eval, _, _, _ = model.rssm.transition(h_eval, z_eval, actions_t)
                z_eval, _, _ = model.rssm.posterior(h_eval, embeds)
                
                # If an env reset happens, we reset its tracking state
                for idx, (term, trunc) in enumerate(zip(terminations, truncations)):
                    if term or trunc:
                        h_eval[idx] = 0.0
                        z_eval[idx] = 0.0

            # Store in sequence replay buffer (we record first env's trajectory to simplify sequential alignment)
            rb.add(obs[0], actions[0], rewards[0], terminations[0] or truncations[0])

            # Record episode returns
            for info in infos.get("final_info", []):
                if info is not None and "episode" in info:
                    eps_ret = float(info["episode"]["r"])
                    eps_len = float(info["episode"]["l"])
                    print(f"global_step={global_step}, episodic_return={eps_ret:.2f}, episodic_length={eps_len:.0f}")
                    writer.add_scalar("charts/episodic_return", eps_ret, global_step)
                    writer.add_scalar("charts/episodic_length", eps_len, global_step)
                    log_mlflow_metrics(mlf_run, {"episodic_return": eps_ret, "episodic_length": eps_len}, global_step)

            obs = next_obs
            global_step += args.num_envs

            # 2. Update World Model and Actor/Critic
            if global_step >= 5000 and rb.size > args.seq_len + 10:
                # Sample a sequence batch (L, B, dim)
                obs_seq, action_seq, reward_seq, done_seq = rb.sample(args.batch_size, args.seq_len)
                
                # Run through world model dynamics
                h_seq, z_seq, prior_means, prior_stds, post_means, post_stds = model(obs_seq, action_seq, done_seq)
                
                # --- World Model Losses ---
                # Symlog target scaling for observations
                target_obs = symlog(obs_seq)
                reconstructed_obs = model.decoder(torch.cat([h_seq, z_seq], dim=-1))
                rec_loss = F.mse_loss(reconstructed_obs, target_obs)
                
                # Reward prediction loss
                predicted_rewards = model.reward_net(torch.cat([h_seq[:-1], z_seq[:-1]], dim=-1))
                reward_loss = F.mse_loss(predicted_rewards, reward_seq)
                
                # Continue prediction loss (predicting terminates)
                predicted_continues = model.continue_net(torch.cat([h_seq[:-1], z_seq[:-1]], dim=-1))
                continue_loss = F.binary_cross_entropy(predicted_continues, 1.0 - done_seq.float())
                
                # KL Divergence regularization loss
                kl_loss = 0.0
                for t in range(len(prior_means)):
                    p_dist = torch.distributions.Normal(prior_means[t], prior_stds[t])
                    q_dist = torch.distributions.Normal(post_means[t], post_stds[t])
                    kl = torch.distributions.kl.kl_divergence(q_dist, p_dist).sum(dim=-1).mean()
                    kl_loss += kl
                kl_loss = kl_loss / len(prior_means)
                
                # Total World Model loss
                model_loss = rec_loss + reward_loss + continue_loss + args.kl_weight * torch.clamp(kl_loss, min=0.1)

                model_opt.zero_grad()
                model_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                model_opt.step()

                # --- Actor-Critic Latent Space Imagination ---
                # Pick a starting batch of states from the sequence
                h_0 = h_seq.detach().view(-1, h_seq.shape[-1])
                z_0 = z_seq.detach().view(-1, z_seq.shape[-1])
                
                # Shuffle and take batch
                indices = torch.randperm(h_0.size(0))[:args.batch_size]
                h_imag = h_0[indices]
                z_imag = z_0[indices]
                
                imag_h, imag_z, imag_actions = [h_imag], [z_imag], []
                imag_rewards, imag_continues = [], []
                
                # Rollout actor inside prior dynamics for H steps
                for _ in range(args.imag_horizon):
                    action = actor.get_action(h_imag, z_imag, sample=True)
                    h_imag, z_imag, _, _ = model.rssm.transition(h_imag, z_imag, action)
                    
                    imag_h.append(h_imag)
                    imag_z.append(z_imag)
                    imag_actions.append(action)
                    
                    state_cat = torch.cat([h_imag, z_imag], dim=-1)
                    r_pred = model.reward_net(state_cat)
                    c_pred = model.continue_net(state_cat)
                    imag_rewards.append(r_pred)
                    imag_continues.append(c_pred)

                imag_h = torch.stack(imag_h, dim=0) # (H+1, B, hidden_dim)
                imag_z = torch.stack(imag_z, dim=0) # (H+1, B, stochastic_dim)
                imag_rewards = torch.stack(imag_rewards, dim=0) # (H, B, 1)
                imag_continues = torch.stack(imag_continues, dim=0) # (H, B, 1)
                
                # Calculate Critic value estimates (detached from dynamics graph)
                values = critic(imag_h.detach(), imag_z.detach()) # (H+1, B, 1)
                
                # Compute targets (GAE/lambda returns in latent space)
                lambda_ = 0.95
                discount = args.gamma * imag_continues
                returns = torch.zeros_like(imag_rewards)
                
                # Detach values for target returns calculation so gradients only flow to actor through dynamics
                detached_values = values.detach()
                last_return = detached_values[-1]
                for t in reversed(range(args.imag_horizon)):
                    returns[t] = imag_rewards[t] + discount[t] * ((1.0 - lambda_) * detached_values[t+1] + lambda_ * last_return)
                    last_return = returns[t]

                # Actor Loss: maximize predicted value (lambda-returns)
                actor_loss = -returns.mean()
                
                actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
                actor_opt.step()
                
                # Critic Loss: minimize MSE with targets (returns is already detached)
                critic_loss = F.mse_loss(values[:-1], returns.detach())
                
                critic_opt.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
                critic_opt.step()

                # Tensorboard/WandB Logs
                if global_step % 1000 < args.num_envs:
                    writer.add_scalar("losses/rec_loss", rec_loss.item(), global_step)
                    writer.add_scalar("losses/reward_loss", reward_loss.item(), global_step)
                    writer.add_scalar("losses/kl_loss", kl_loss.item(), global_step)
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                    writer.add_scalar("losses/critic_loss", critic_loss.item(), global_step)
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                        }, step=global_step)
                    log_mlflow_metrics(mlf_run, {
                        "rec_loss": rec_loss.item(),
                        "reward_loss": reward_loss.item(),
                        "kl_loss": kl_loss.item(),
                        "actor_loss": actor_loss.item(),
                        "critic_loss": critic_loss.item(),
                        "sps": int(global_step / (time.time() - start_time)),
                    }, global_step)

            # Checkpoint saving
            if global_step >= next_checkpoint_step:
                ckpt_path = os.path.join(ckpt_dir, f"dreamer_ckpt_{global_step}.pt")
                save_dreamer_checkpoint(
                    ckpt_path,
                    global_step,
                    model,
                    actor,
                    critic,
                    model_opt,
                    actor_opt,
                    critic_opt,
                    envs,
                    args.task_phase,
                    args.target_forward_velocity,
                )
                log_mlflow_artifact(mlf_run, ckpt_path, "dreamer", global_step)
                next_checkpoint_step += args.checkpoint_interval

    except KeyboardInterrupt:
        print("\n[TRAIN] Interrupted by user. Saving checkpoint...")
    finally:
        final_checkpoint_path = os.path.join(ckpt_dir, f"dreamer_ckpt_{global_step}.pt")
        save_dreamer_checkpoint(
            final_checkpoint_path,
            global_step,
            model,
            actor,
            critic,
            model_opt,
            actor_opt,
            critic_opt,
            envs,
            args.task_phase,
            args.target_forward_velocity,
        )
        log_mlflow_artifact(mlf_run, final_checkpoint_path, "dreamer", global_step)
        end_mlflow_run(mlf_run)
        envs.close()
        writer.close()
        print(f"Training completed. Total steps: {global_step}")

if __name__ == "__main__":
    train_dreamer()
