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
    force_delete_run,
    get_checkpoint_dir,
    get_rng_state,
    set_rng_state,
    wrap_normalize_observation,
    wrap_transform_observation,
    ENV_VERSION,
    PHYSICS_PRESETS,
    env_version_of,
    start_mlflow_run,
    log_mlflow_metrics,
    log_mlflow_artifact,
    end_mlflow_run,
)
from envs.reward_shaping import TRAINING_REWARD_KWARGS

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
    """Replay buffer holding one contiguous stream of transitions per environment.

    Sequences are sampled as contiguous windows, so each environment needs an unbroken
    run of rows. The first version stored a single interleaved stream and therefore only
    ever recorded `obs[0]` - which silently threw away num_envs-1 of every collected
    batch while the training budget counted all of them. Adding all envs to that flat
    stream would have spliced unrelated episodes into one "sequence", so the storage is
    now per-environment: the same sample semantics, but every transition is used.
    """

    def __init__(self, capacity, num_envs, obs_shape, action_shape, device):
        self.num_envs = max(1, int(num_envs))
        self.horizon = max(2, int(capacity) // self.num_envs)
        self.capacity = self.horizon * self.num_envs
        self.device = device

        self.obs = np.zeros((self.num_envs, self.horizon, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.num_envs, self.horizon, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((self.num_envs, self.horizon, 1), dtype=np.float32)
        self.dones = np.zeros((self.num_envs, self.horizon, 1), dtype=np.bool_)

        self.idx = 0        # column that the next add() writes
        self.filled = 0     # columns written so far (0..horizon)
        self.size = 0       # total transitions, kept for logging/compat

    def add(self, obs, action, reward, done):
        """Append one column of transitions for every environment at once."""
        self.obs[:, self.idx] = obs
        self.actions[:, self.idx] = action
        self.rewards[:, self.idx] = np.asarray(reward).reshape(-1, 1)
        self.dones[:, self.idx] = np.asarray(done).reshape(-1, 1)

        self.idx = (self.idx + 1) % self.horizon
        self.filled = min(self.filled + 1, self.horizon)
        self.size = min(self.size + self.num_envs, self.capacity)

    def sample(self, batch_size, seq_len):
        # We need to sample sequences of length `seq_len`
        obs_batch, action_batch, reward_batch, done_batch = [], [], [], []

        for _ in range(batch_size):
            # Pick a random environment and a window inside its written region
            valid = False
            while not valid:
                env_no = int(np.random.randint(self.num_envs))
                if self.filled <= seq_len:
                    start, end = 0, max(2, self.filled)
                else:
                    start = int(np.random.randint(0, self.filled - seq_len))
                    end = start + seq_len
                # Skip windows that wrap over the write head
                if start <= self.idx < end:
                    continue
                valid = True

            obs_batch.append(self.obs[env_no, start:end])
            action_batch.append(self.actions[env_no, start:end - 1])
            reward_batch.append(self.rewards[env_no, start:end - 1])
            done_batch.append(self.dones[env_no, start:end - 1])

            
        # Transpose batches to match sequence-first shape (L, B, dim)
        obs_t = torch.as_tensor(np.array(obs_batch), device=self.device).permute(1, 0, 2)
        action_t = torch.as_tensor(np.array(action_batch), device=self.device).permute(1, 0, 2)
        reward_t = torch.as_tensor(np.array(reward_batch), device=self.device)
        done_t = torch.as_tensor(np.array(done_batch), device=self.device)
        
        return obs_t, action_t, reward_t.permute(1, 0, 2), done_t.permute(1, 0, 2)

def imagination_heads(state_cat, model, fused, impl):
    """The two heads that read `[h, z]`, either as the separate modules or as one fused pass.

    Kept as two implementations of the same function rather than one, so the A/B in
    `bench_dreamer_update.py --mode heads-ab` can price the fusion inside a single window instead of
    comparing runs taken hours apart - the same reason `train_redq.py` still carries
    `--ensemble-impl loop`.
    """
    if impl == "separate":
        # continue_net is a Sequential that already ends in Sigmoid, so it returns a probability:
        # wrapping it in another torch.sigmoid here is the bug the golden-loss test caught.
        return model.reward_net(state_cat), model.continue_net(state_cat)
    w1, b1, w2, b2 = fused
    heads = F.linear(F.elu(F.linear(state_cat, w1, b1)), w2, b2)
    return heads[:, 0:1], torch.sigmoid(heads[:, 1:2])


def fused_heads(reward_net, continue_net):
    """One pass for the two heads that read the same `[h, z]`: 4 kernels per step instead of 6.

    The merged first layer is the two trunks side by side, and the merged output layer is
    block-diagonal, so each output still depends only on its own trunk half - the other half
    multiplies zeros, which doubles *that layer's* arithmetic to remove two launches. That trade is
    the point: a captured update spends 0.8% of its time on arithmetic
    (benchmarks/dreamer_update_scaling.json against the 15-20 TFLOP/s this card measures), so
    launches are what costs.

    Rebuilt once per update rather than cached, because the weights move between updates and a
    captured graph replays whatever tensors it recorded. Gradients still flow into the original
    `reward_net` / `continue_net` parameters, so the checkpoint format and `model_opt` are
    untouched.
    """
    w1 = torch.cat([reward_net[0].weight, continue_net[0].weight], dim=0)
    b1 = torch.cat([reward_net[0].bias, continue_net[0].bias], dim=0)
    rw, rb = reward_net[2].weight, reward_net[2].bias
    cw, cb = continue_net[2].weight, continue_net[2].bias
    z = torch.zeros_like(rw)
    w2 = torch.cat([torch.cat([rw, z], dim=1), torch.cat([z, cw], dim=1)], dim=0)
    b2 = torch.cat([rb, cb], dim=0)
    return w1, b1, w2, b2


def dreamer_update(model, actor, critic, model_opt, actor_opt, critic_opt, batch, args,
                   start_indices, zero_grad_set_to_none=True):
    """One Dreamer update: fit the world model on a sequence batch, then train on imagination.

    Moved out of the training loop unchanged, so that the update can be timed, tested and captured
    without an environment in front of it. `start_indices` picks which latents start the rollout;
    the caller draws it because a captured CUDA graph cannot reach the CPU RNG, and no other
    torch-CPU RNG consumer sits between the replay sample and this call, so hoisting it one line
    keeps the indices. `zero_grad_set_to_none=False` is what capture needs (an allocation per
    parameter per replay is not replayable); it is numerically identical here because every
    parameter of all three optimizers receives a gradient in this update.
    """
    obs_seq, action_seq, reward_seq, done_seq = batch

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

    # KL Divergence regularization loss.
    # One batched pair of Normal distributions instead of 49 constructions: the
    # loop cost 64.9 ms of a 289 ms update step - as much as the whole RSSM
    # forward pass - while the stacked form measures 0.78 ms for the same value
    # (0.492967 against 0.492967, 6e-8 relative, pure float reassociation).
    #
    # The pairing is also fixed here. prior_means[t] is p(z_{t+1} | h_{t+1}),
    # produced inside loop iteration t of WorldModel.forward, while post_means[t]
    # is q(z_t | h_t, x_t) - post_means[0] being the pre-action posterior at step
    # 0. Indexing both with the same t compared the prediction of one step with
    # the posterior of the previous one, so the regularizer was trained against a
    # shifted target; [1:1+T] puts prior[t] against post[t+1].
    T = len(prior_means)
    # validate_args=False: the checks compare on the host, which is a synchronisation point, and a
    # synchronisation point cannot be captured. They gate nothing here - the stds are softplus +
    # 0.1 by construction - and skipping them changes no value.
    p_dist = torch.distributions.Normal(
        torch.stack(prior_means), torch.stack(prior_stds), validate_args=False)
    q_dist = torch.distributions.Normal(
        torch.stack(post_means[1:1 + T]), torch.stack(post_stds[1:1 + T]), validate_args=False)
    kl_loss = torch.distributions.kl.kl_divergence(q_dist, p_dist).sum(dim=-1).mean()

    # Total World Model loss
    model_loss = rec_loss + reward_loss + continue_loss + args.kl_weight * torch.clamp(kl_loss, min=0.1)

    model_opt.zero_grad(set_to_none=zero_grad_set_to_none)
    model_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    model_opt.step()

    # --- Actor-Critic Latent Space Imagination ---
    # Pick a starting batch of states from the sequence
    h_0 = h_seq.detach().view(-1, h_seq.shape[-1])
    z_0 = z_seq.detach().view(-1, z_seq.shape[-1])

    h_imag = h_0[start_indices]
    z_imag = z_0[start_indices]

    # The two heads that read [h, z] are evaluated as one pass for the whole rollout; see
    # fused_heads - rebuilt here, once, so the loop below stays launch-cheap.
    impl = getattr(args, "heads_impl", "fused")   # the shipped default; see --heads-impl
    fused = fused_heads(model.reward_net, model.continue_net)

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
        r_pred, c_pred = imagination_heads(state_cat, model, fused, impl)
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

    actor_opt.zero_grad(set_to_none=zero_grad_set_to_none)
    actor_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
    actor_opt.step()

    # Critic Loss: minimize MSE with targets (returns is already detached)
    critic_loss = F.mse_loss(values[:-1], returns.detach())

    critic_opt.zero_grad(set_to_none=zero_grad_set_to_none)
    critic_loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
    critic_opt.step()

    return {"rec_loss": rec_loss, "reward_loss": reward_loss, "continue_loss": continue_loss,
            "kl_loss": kl_loss, "actor_loss": actor_loss, "critic_loss": critic_loss}


def latest_dreamer_checkpoint(ckpt_dir):
    """The highest-step Dreamer checkpoint in `ckpt_dir`, or None. Shared by the resume guard and
    the update-mode resolution below, which have to agree on which file they are talking about."""
    if not os.path.isdir(ckpt_dir):
        return None
    files = [f for f in os.listdir(ckpt_dir)
             if f.startswith("dreamer_ckpt_") and f.endswith(".pt")]
    if not files:
        return None
    files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return os.path.join(ckpt_dir, files[-1])


def resolve_update_graph(requested, device_type, resumed_graph=None):
    """Decide whether the Dreamer update runs captured, and say why, in one testable place.

    Three inputs, in priority order: an explicit flag from the user, the configuration of the
    checkpoint being resumed, and the device. The middle one exists because the default flipped to
    "capture when CUDA is present" - without it, every pre-existing eager run would refuse to
    resume the day this ships, since Adam's state is laid out differently between the two.
    """
    if requested is not None:
        if requested and device_type != "cuda":
            raise SystemExit("--update-graph replays a captured CUDA graph, so it needs a CUDA device.")
        return requested
    if resumed_graph is not None:
        return resumed_graph
    return device_type == "cuda"


def make_adam(params, lr, graph):
    """Adam, in the one configuration CUDA graph capture accepts when the update is captured.

    `capturable=True` moves the step counter onto the device so a replay advances it instead of
    reading a frozen host value; that mode requires `foreach=False`, and the unfused path is worth
    about 1e-6 on the weights against the fused one (measured, tests/test_dreamer_graph.py). The
    eager default stays exactly as it always was.
    """
    if graph:
        return optim.Adam(params, lr=lr, capturable=True, foreach=False)
    return optim.Adam(params, lr=lr)


class CapturedDreamerUpdate:
    """The whole Dreamer update as one CUDA graph replay.

    Why this rather than stacking tensors the way `BatchedSoftQEnsemble` does: one update walks 49
    timesteps of world model plus `--imag-horizon` of imagination, and h_t depends on h_{t-1}, so
    there is nothing to stack. What there is instead is dispatch cost - about 14,400 aten calls per
    update on this box (benchmarks/dreamer_update_profile.json) against six 256-unit MLPs, so the
    GPU spends the step waiting for Python. A graph records those calls once and replays them as a
    single launch.

    Capture is strict about what it records, and all of its rules are what the constructor sets up:
    static shapes (the replay buffer always yields (seq_len, batch_size, dim)), static input buffers
    that the new minibatch is copied into, no host tensor or synchronisation inside (hence
    `start_indices` from the caller, `foreach=True` off, `set_to_none=False`, `validate_args=False`),
    and a side-stream warmup that also lets Adam allocate its state before the capture owns it.
    """

    def __init__(self, fn, batch, start_indices, warmup=3):
        self.static = [t.clone() for t in batch]
        self.static_indices = start_indices.to(self.static[0].device).clone()
        call = lambda: fn(tuple(self.static), self.static_indices)

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                call()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.outputs = call()

    def __call__(self, batch, start_indices):
        for buffer, source in zip(self.static, batch):
            buffer.copy_(source)
        self.static_indices.copy_(start_indices)
        self.graph.replay()
        return self.outputs


def advance_recurrent_state(model, h, z, action, next_obs, terminations, truncations):
    """Advance the collection lanes' recurrent state by one environment step.

    A function rather than inline because it is three GPU round-trips plus a Python reset loop per
    collection step, and it sits outside the captured update: `bench_dreamer_update.py --mode
    loop-split` has to price the shipped path to say how much of an iteration the collection side is.
    """
    device = h.device
    with torch.no_grad():
        actions_t = torch.as_tensor(action, dtype=torch.float32, device=device)
        embeds = model.encoder(torch.as_tensor(next_obs, dtype=torch.float32, device=device))
        h, _, _, _ = model.rssm.transition(h, z, actions_t)
        z, _, _ = model.rssm.posterior(h, embeds)

        # If an env reset happens, we reset its tracking state
        for idx, (term, trunc) in enumerate(zip(terminations, truncations)):
            if term or trunc:
                h[idx] = 0.0
                z[idx] = 0.0
    return h, z


def parse_dreamer_args():
    parser = argparse.ArgumentParser(description="DreamerV3 Walker Ragdoll Training")
    parser.add_argument("--run-id", type=str, default="walker_dreamer_1m")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--total-timesteps", type=int, default=1000000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--learning-starts", type=int, default=5000,
                        help="Env steps before the first update. Was hardcoded at 5000; the flag "
                             "exists so a test can reach the update path without a 5k-step budget.")
    parser.add_argument("--num-envs", type=int, default=4) # Smaller number of envs since sequential data logging is seq-based
    parser.add_argument("--buffer-size", type=int, default=200000)
    parser.add_argument("--gamma", type=float, default=0.99)
    # 50k, not the 200k this started at: a full checkpoint measures 7.3 MB and takes 15.8 ms to
    # write (timed on the real architectures), so 20 saves per 1M steps cost 0.32 s and 146 MB,
    # while a 200k interval is what left the 1M-step run that died at 84,456 steps with nothing to
    # resume from.
    # 50k, not the 200k this started at. One full checkpoint is 6.96 MiB and 22.16 ms to write
    # (benchmarks/dreamer_checkpoint_cost.json, `python bench_dreamer_update.py --mode
    # checkpoint-cost`), so 20 saves per 1M steps cost 0.44 s and 139.3 MiB - while 200k was
    # exactly the gap that left the run that died at 84,456 steps with nothing to resume from.
    parser.add_argument("--checkpoint-interval", type=int, default=50000)
    parser.add_argument("--heads-impl", choices=["fused", "separate"], default="fused",
                        help="reward/continue heads in the imagination rollout: one fused pass "
                             "(default) or the two modules. The fused path measured 1.6-3.8%% "
                             "faster on the whole update across three paired same-process runs "
                             "(benchmarks/dreamer_heads_ab.json) and is equivalent to ~1e-5 on the "
                             "losses and the parameter gradients; `separate` reproduces the "
                             "pre-fusion numerics exactly, which is what the golden-loss test pins.")
    parser.add_argument("--reset-mode", type=str, default="upright")
    parser.add_argument("--fixed-reset-probability", type=float, default=0.25)
    parser.add_argument("--upright-reset-probability", type=float, default=0.15)
    parser.add_argument("--fallen-velocity-scale", type=float, default=0.35)
    parser.add_argument("--task-phase", type=str, default="target")
    parser.add_argument("--target-forward-velocity", type=float, default=0.8)
    # Declared here as well as in train_walker: build_vec_env reads the attribute, and a trainer that
    # does not define it would silently collect in v9 while the flag says otherwise on the CLI.
    parser.add_argument("--physics-preset", choices=PHYSICS_PRESETS, default="v9",
                       help="which compiled world to collect in; 'v9' is the published one")
    add_vec_env_args(parser)
    add_device_arg(parser)
    
    # DreamerV3 specific hyperparameters
    parser.add_argument("--seq-len", type=int, default=50, help="World model sequence batch training length (L)")
    parser.add_argument("--imag-horizon", type=int, default=15, help="Imagination sequence length (H)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size of sequences")
    parser.add_argument("--kl-weight", type=float, default=1.0, help="KL divergence regularization weight")
    parser.add_argument("--update-graph", action=argparse.BooleanOptionalAction, default=None,
                        help="capture the whole update as one CUDA graph. Unset means auto: on for "
                             "CUDA, off elsewhere, and on resume it follows the configuration the "
                             "checkpoint was written in (see resolve_update_graph).")
    
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
        # Recorded so an evaluator scores this policy under the reward it was trained with. Without
        # it the scorer has to guess, and it guessed wrong once already (see reward_kwargs_for).
        "reward_kwargs": dict(TRAINING_REWARD_KWARGS),
    }, actor_only_path)
    
    # Save full state
    state = {
        "algo": "dreamer",
        "env_version": env_version_of(envs),
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
        "reward_kwargs": dict(TRAINING_REWARD_KWARGS),
        "rng_state": get_rng_state(),
    }
    torch.save(state, ckpt_path)
    print(f"[CHECKPOINT] Saved at step {global_step} -> {ckpt_path}")

def train_dreamer():
    args = parse_dreamer_args()
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
    # Resolve the update mode before the optimizers exist: Adam's state layout is the thing that
    # differs between the two, so the choice has to be made once, here, and every later check
    # compares against what was actually resolved.
    ckpt_dir = get_checkpoint_dir(args.run_id)
    requested = args.update_graph
    resume_ckpt = latest_dreamer_checkpoint(ckpt_dir) if args.resume else None
    # The mode is read out of the checkpoint, so the file has to be opened before the optimizers
    # are built. This is the same read the resume path always did, moved up rather than widened:
    # it happens only under --resume, only on checkpoints this trainer wrote, and these files carry
    # obs_rms / rng_state objects, so a weights-only read cannot open them at all.
    checkpoint = None
    if resume_ckpt:
        print(f"[CHECKPOINT] Loading from {resume_ckpt}")
        checkpoint = torch.load(resume_ckpt, map_location=device, weights_only=False)
    resumed_graph = (
        bool(checkpoint["model_opt_state_dict"]["param_groups"][0].get("capturable"))
        if checkpoint is not None else None)
    args.update_graph = resolve_update_graph(requested, device.type, resumed_graph)
    mode = "captured CUDA graph" if args.update_graph else "eager"
    source = ("--update-graph" if requested else "--no-update-graph") if requested is not None else \
        ("follows " + os.path.basename(resume_ckpt)) if resumed_graph is not None else \
        f"auto for device={device.type}"
    print(f"[DREAMER] update: {mode} ({source})")

    envs = build_vec_env(args, run_name, capture_video=False)
    envs = wrap_normalize_observation(envs)
    envs = wrap_transform_observation(envs, lambda obs: np.clip(obs, -10, 10))

    obs_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(np.prod(envs.single_action_space.shape))

    # Define Dreamer architectures
    model = WorldModel(obs_dim, action_dim).to(device)
    actor = DreamerActor(hidden_dim=256, stochastic_dim=32, action_dim=action_dim).to(device)
    critic = DreamerCritic(hidden_dim=256, stochastic_dim=32).to(device)

    # Optimizers
    model_opt = make_adam(model.parameters(), args.learning_rate, args.update_graph)
    actor_opt = make_adam(actor.parameters(), args.learning_rate, args.update_graph)
    critic_opt = make_adam(critic.parameters(), args.learning_rate, args.update_graph)

    rb = SequenceReplayBuffer(args.buffer_size, args.num_envs, envs.single_observation_space.shape, envs.single_action_space.shape, device)
    global_step = 0
    restored_obs_rms = None

    # Setup WandB (removido: tracking canônico é MLflow + TensorBoard; wandb não é dependência)
    # Setup MLflow
    mlf_run = start_mlflow_run(args, run_name, "dreamer",
                               extra_params={"update_graph": args.update_graph})

    # Resume handling (the file was already opened above, to settle the update mode)
    if checkpoint is not None:
        ckpt_path = resume_ckpt
        ckpt_graph = resumed_graph
        if ckpt_graph != args.update_graph:
            raise SystemExit(
                f"[CHECKPOINT] {ckpt_path} was written with "
                f"{'the captured update' if ckpt_graph else 'the eager update'}, and Adam's state is "
                "laid out differently between the two (the capturable one keeps its step counter on "
                "the device), so resuming across them is refused rather than half-loaded. Pass "
                f"{'--update-graph' if ckpt_graph else '--no-update-graph'} to continue this run.")
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
    captured_update = None

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
            h_eval, z_eval = advance_recurrent_state(model, h_eval, z_eval, actions, next_obs,
                                                      terminations, truncations)

            # Store in sequence replay buffer (one column per env; each env keeps its own
            # contiguous stream so sampled sequences never splice two episodes together)
            rb.add(obs, actions, rewards, np.logical_or(terminations, truncations))

            # Record episode returns
            for info in infos.get("final_info", []):
                if info is not None and "episode" in info:
                    eps_ret = float(np.asarray(info["episode"]["r"]).item())
                    eps_len = float(np.asarray(info["episode"]["l"]).item())
                    print(f"global_step={global_step}, episodic_return={eps_ret:.2f}, episodic_length={eps_len:.0f}")
                    writer.add_scalar("charts/episodic_return", eps_ret, global_step)
                    writer.add_scalar("charts/episodic_length", eps_len, global_step)
                    log_mlflow_metrics(mlf_run, {"episodic_return": eps_ret, "episodic_length": eps_len}, global_step)

            obs = next_obs
            global_step += args.num_envs

            # 2. Update World Model and Actor/Critic
            # A sequence needs seq_len contiguous rows inside ONE env's stream, so the
            # gate is on rows written per env (rb.filled), not total transitions.
            if global_step >= args.learning_starts and rb.filled > args.seq_len + 10:
                # Sample a sequence batch (L, B, dim)
                batch = rb.sample(args.batch_size, args.seq_len)
                # The imagination rollout's starting latents, drawn here because a captured
                # CUDA graph cannot call the CPU RNG. Nothing between this and the old call site
                # consumes torch's CPU generator, so the indices are the same ones as before.
                start = torch.randperm(args.seq_len * args.batch_size)[:args.batch_size]
                if args.update_graph:
                    if captured_update is None:
                        captured_update = CapturedDreamerUpdate(
                            lambda b, i: dreamer_update(model, actor, critic, model_opt, actor_opt,
                                                        critic_opt, b, args, i,
                                                        zero_grad_set_to_none=False),
                            batch, start)
                    losses = captured_update(batch, start)
                else:
                    losses = dreamer_update(model, actor, critic, model_opt, actor_opt, critic_opt,
                                            batch, args, start)

                # Tensorboard/WandB Logs
                if global_step % 1000 < args.num_envs:
                    writer.add_scalar("losses/rec_loss", losses["rec_loss"].item(), global_step)
                    writer.add_scalar("losses/reward_loss", losses["reward_loss"].item(), global_step)
                    writer.add_scalar("losses/kl_loss", losses["kl_loss"].item(), global_step)
                    writer.add_scalar("losses/actor_loss", losses["actor_loss"].item(), global_step)
                    writer.add_scalar("losses/critic_loss", losses["critic_loss"].item(), global_step)
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    log_mlflow_metrics(mlf_run, {
                        "rec_loss": losses["rec_loss"].item(),
                        "reward_loss": losses["reward_loss"].item(),
                        "kl_loss": losses["kl_loss"].item(),
                        "actor_loss": losses["actor_loss"].item(),
                        "critic_loss": losses["critic_loss"].item(),
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
