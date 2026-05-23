# Walker Ragdoll - SAC / PPO / TD3 with CleanRL and MuJoCo

Walker ragdoll training project using SAC (Soft Actor-Critic), PPO (Proximal Policy Optimization), and TD3 (Twin Delayed DDPG) with MuJoCo and a CleanRL-style environment.

## Project Structure

```
.
├── train_walker.py            # Main training script (SAC + PPO + TD3)
├── train_redq.py              # REDQ training script (Random Ensemble Double Q-learning)
├── train_ars.py               # ARS training script (Augmented Random Search)
├── train_dreamer.py           # DreamerV3 training script (Model-based RSSM/Imagination)
├── play.py                    # Real-time visualization of a trained agent (SAC, PPO, etc.)
├── play_race.py               # Multi-agent visual and headless evaluation race
├── walker_ragdoll.xml         # MuJoCo model for the walker ragdoll
├── envs/
│   └── walker_ragdoll_env.py # Custom Gymnasium environment
├── tests/
│   ├── test_env.py            # Environment unit tests
│   ├── test_training.py       # Training integration tests (SAC + PPO + TD3)
│   └── test_race.py           # Multi-agent race system tests
├── requirements.txt           # Project dependencies
├── .env.example               # Environment variables example
├── .env                       # Default environment variables
└── README.md                  # This file
```

## Installation

1. Install dependencies:
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. Verify MuJoCo is installed correctly:
```bash
python -c "import mujoco; print(mujoco.__version__)"
```

## Docker

You can also run the project using Docker:

### Build
```bash
docker build -t walker-marl .
```

### Run Training
```bash
docker run -v ${PWD}/checkpoints:/app/checkpoints -v ${PWD}/runs:/app/runs walker-marl
```

## Usage

### Algorithm selection

Use `--algo sac`, `--algo ppo`, or `--algo td3` to select the training algorithm in the main script, or run the specialized scripts for the new algorithms:

```bash
# SAC training (default)
python train_walker.py --algo sac --run-id walker_sac_v1

# PPO training
python train_walker.py --algo ppo --run-id walker_ppo_v1

# TD3 training
python train_walker.py --algo td3 --run-id walker_td3_v1

# REDQ training (Random Ensemble Double Q-learning)
python train_redq.py --run-id walker_redq_v1

# ARS training (Augmented Random Search)
python train_ars.py --run-id walker_ars_v1

# DreamerV3 training (Model-based RL)
python train_dreamer.py --run-id walker_dreamer_v1
```

### Experiment Tracking

Every training run is tracked:

#### MLflow
All runs are logged to MLflow under the `walker-ragdoll` experiment (or whatever name is in the `MLFLOW_EXPERIMENT` environment variable).
- Hyperparameters are stored as MLflow **params**.
- Training metrics (episodic return, losses, SPS) are stored as MLflow **metrics** with step numbers.
- Checkpoint files are logged as MLflow **artifacts**.
- MLflow data is written locally to the `mlruns/` directory by default.

Launch the MLflow UI to browse all runs:
```bash
mlflow ui
# then open http://localhost:5000
```

To point MLflow at a remote tracking server instead:
```bash
export MLFLOW_TRACKING_URI=http://my-mlflow-server:5000
```

### Multi-Agent Race Evaluation
You can evaluate and race different trained agents (even using different algorithms and architectures) in parallel lanes in the same MuJoCo simulation environment using the `play_race.py` script.

#### Command Example
```bash
python play_race.py --checkpoints checkpoints/walker_sac_v1 checkpoints/walker_ppo_v1 checkpoints/walker_td3_v1 --names AgentSAC AgentPPO AgentTD3 --target-x 15.0
```

#### Race Options:
- `--checkpoints`: List of checkpoints (files or folders). Folders will automatically resolve to the latest checkpoint in that folder.
- `--names`: Optional display names for each competitor.
- `--target-x`: The distance (in meters) to reach the finish line (default: `15.0`).
- `--lane-distance`: Spacing (in meters) between lanes (default: `1.8`).
- `--headless`: Run the simulation in console/headless mode (printing positions and final standings table) without launching the GUI.
- `--max-steps`: Maximum simulation steps (default: `2500`).

The visual camera tracks the current leading walker dynamically throughout the race!

### Unity ML-Agents-style features

- **Automatic checkpointing**: Saves checkpoints at each configured interval.
- **Resume**: Continue training from where it stopped (restoring replay buffers and RNG states).
- **Force**: Delete previous data and start from scratch.
- **Run ID**: Unique identifier for each experiment.

#### Examples

**Start a new training run:**
```bash
python train_walker.py --run-id walker_v1 --seed 1 --total-timesteps 1000000
```

**Resume training:**
```bash
python train_walker.py --run-id walker_v1 --resume
```

**Force restart (deletes previous checkpoints/logs):**
```bash
python train_walker.py --run-id walker_v1 --force --seed 2
```

## Hyperparameters

All hyperparameters can be configured via command line or environment variables.

### Shared Parameters

| CLI Parameter | Environment Variable | Default | Description |
|---|---|---|---|
| `--algo` | `ALGO` | `sac` | Training algorithm: `sac`, `ppo`, or `td3` |
| `--run-id` | `RUN_ID` | `walker_train` | Experiment ID |
| `--seed` | `SEED` | `1` | Random seed |
| `--checkpoint-interval` | `CHECKPOINT_INTERVAL` | `1000000` | Checkpoint interval in timesteps |
| `--total-timesteps` | `TOTAL_TIMESTEPS` | `1000000` | Total timesteps |
| `--learning-rate` | `LEARNING_RATE` | `3e-4` | Learning rate |
| `--num-envs` | `NUM_ENVS` | `128` | Number of parallel environments |
| `--gamma` | `GAMMA` | `0.99` | Discount factor |
| `--reset-mode` | `RESET_MODE` | `mixed` | Initial-state distribution: `fixed`, `mixed`, `fallen`, or `upright` |
| `--fixed-reset-probability` | `FIXED_RESET_PROBABILITY` | `0.25` | In mixed mode, fraction of episodes using the old fixed fallen pose |
| `--upright-reset-probability` | `UPRIGHT_RESET_PROBABILITY` | `0.15` | In mixed mode, fraction of episodes starting almost upright |
| `--fallen-velocity-scale` | `FALLEN_VELOCITY_SCALE` | `0.35` | Extra velocity noise for randomized fallen resets |
| `--task-phase` | `TASK_PHASE` | `recovery` | Reward curriculum phase: `recovery`, `balance`, `walk`, or `target` |
| `--target-forward-velocity` | `TARGET_FORWARD_VELOCITY` | `0.8` | Target x velocity for the walk phase |

### SAC-specific Parameters

| CLI Parameter | Environment Variable | Default | Description |
|---|---|---|---|
| `--buffer-size` | `BUFFER_SIZE` | `1000000` | Replay buffer size |
| `--batch-size` | `BATCH_SIZE` | `256` | SAC batch size |
| `--learning-starts` | `LEARNING_STARTS` | `10000` | Random exploration steps before updates |
| `--tau` | `TAU` | `0.005` | Target network update rate |
| `--alpha` | `ALPHA` | `0.2` | Initial entropy temperature |
| `--init-from-run-id` | `INIT_FROM_RUN_ID` | - | Start a new run from another run's actor weights |
| `--save-replay-buffer` / `--no-save-replay-buffer` | `SAVE_REPLAY_BUFFER` | `true` | Include or skip the replay buffer in checkpoints |

### PPO-specific Parameters

| CLI Parameter | Environment Variable | Default | Description |
|---|---|---|---|
| `--num-steps` | `NUM_STEPS` | `2048` | Rollout length per environment per update |
| `--num-minibatches` | `NUM_MINIBATCHES` | `32` | Number of minibatches per update |
| `--update-epochs` | `UPDATE_EPOCHS` | `10` | Number of optimization epochs per rollout |
| `--clip-coef` | `CLIP_COEF` | `0.2` | PPO clip coefficient |
| `--ent-coef` | `ENT_COEF` | `0.0` | Entropy bonus coefficient |
| `--vf-coef` | `VF_COEF` | `0.5` | Value function loss coefficient |
| `--gae-lambda` | `GAE_LAMBDA` | `0.95` | GAE lambda for advantage estimation |
| `--max-grad-norm` | `MAX_GRAD_NORM` | `0.5` | Maximum gradient norm for clipping |

### TD3-specific Parameters

| CLI Parameter | Environment Variable | Default | Description |
|---|---|---|---|
| `--buffer-size` | `BUFFER_SIZE` | `1000000` | Replay buffer size |
| `--batch-size` | `BATCH_SIZE` | `256` | TD3 batch size |
| `--learning-starts` | `LEARNING_STARTS` | `10000` | Random exploration steps before updates |
| `--tau` | `TAU` | `0.005` | Target network update rate |
| `--exploration-noise` | `EXPLORATION_NOISE` | `0.1` | Action exploration noise std |
| `--policy-noise` | `POLICY_NOISE` | `0.2` | Target action smoothing noise std |
| `--noise-clip` | `NOISE_CLIP` | `0.5` | Target action smoothing noise clip range |
| `--policy-frequency` | `POLICY_FREQUENCY` | `2` | Delay parameter for policy updates |

## Training Recommendations

For the `WalkerRagdoll-v0` environment, the reward is split into phases. `recovery` keeps the stand-up objective, `balance` adds standing stability, foot-only support, low drift, and low torso velocity, `walk` adds gated forward-velocity tracking after the agent is upright, and `target` adds a sampled navigation target with reward for reducing distance while standing. In the `target` phase, the environment uses curriculum learning: the target only respawns at a new random position after the agent reaches it 10 consecutive times.

Recommended phase progression:

```bash
# Phase 1: learn to stand up
python train_walker.py --run-id walker_recovery_v1 --force --total-timesteps 20000000

# Phase 2: learn to stay upright
python train_walker.py --run-id walker_balance_v1 --task-phase balance --init-from-run-id walker_recovery_v1 --total-timesteps 30000000

# Phase 3: start walking
python train_walker.py --run-id walker_walk_v1 --task-phase walk --init-from-run-id walker_balance_v1 --total-timesteps 50000000

# Phase 4: walk to sampled targets
python train_walker.py --run-id walker_target_v1 --task-phase target --init-from-run-id walker_walk_v1 --total-timesteps 70000000 --no-save-replay-buffer
```

PPO training example:

```bash
python train_walker.py --algo ppo --run-id walker_ppo_v1 --task-phase recovery --total-timesteps 20000000 --num-envs 64
```

## Recovery-Supervised Target Training Pipeline

We provide a script `run_training_pipeline.py` to train SAC, PPO, and TD3 agents sequentially for 1M steps in the `target` task phase. The training integrates a **Recovery Supervisor** which intercepts falling states (torso height $z < 1.1$ or upright factor $< 0.8$) and yields control to a pre-trained recovery policy (`walker_recovery_v1`). The training experiences collected under recovery control are excluded from the target agents' update computations via buffer/loss masking.

To run the full 1M-step sequential pipeline:
```bash
python run_training_pipeline.py
```
This will sequentially produce runs `walker_sac_1m`, `walker_ppo_1m`, and `walker_td3_1m`, logging all training metrics locally to offline W&B.

## Tests

Run unit and integration tests:
```bash
python -m unittest discover -s tests -v
```

Or individually:
```bash
python -m tests.test_env
python -m tests.test_training
python -m tests.test_race
```

## TensorBoard

View training logs:
```bash
tensorboard --logdir runs/
```

## Visualizing the Trained Agent

Use the `play.py` script to watch the trained agent in real time. It automatically detects whether the checkpoint is SAC or PPO.

### Watch live (opens a MuJoCo window, runs until you press Ctrl+C)
```bash
python play.py --run-id walker_v1 --render-mode human
```

### Run a fixed number of episodes
```bash
python play.py --run-id walker_v1 --num-episodes 10 --render-mode human
```

### Stochastic mode (sample from the policy)
```bash
python play.py --run-id walker_v1 --stochastic --render-mode human
```

### Load a specific checkpoint step
```bash
python play.py --run-id walker_v1 --checkpoint-step 500000 --render-mode human
```

**Note:** `play.py` loads the saved observation normalization statistics and freezes them during evaluation.

### Watch an old PPO legacy checkpoint
```bash
python play_ppo_legacy.py --run-id Humanoid_Curriculum_v1 --checkpoint-step 133801920 --render-mode human --deterministic
```

## Checkpoints

SAC checkpoints are saved to `checkpoints/<run_id>/sac_ckpt_<step>.pt` and include the actor, critics, entropy temperature, optimizers, and observation normalization statistics. By default they also include the replay buffer. Use `--no-save-replay-buffer` for lighter checkpoints.

PPO checkpoints are saved to `checkpoints/<run_id>/ppo_ckpt_<step>.pt` and include the full actor-critic agent, optimizer, and observation normalization statistics. PPO checkpoints are always lightweight since there is no replay buffer.

TD3 checkpoints are saved to `checkpoints/<run_id>/td3_ckpt_<step>.pt` and include the actor, both critics, the target actor/critics, optimizers, and observation normalization statistics.

REDQ checkpoints are saved to `checkpoints/<run_id>/redq_ckpt_<step>.pt` and include the actor, ensemble of critics, target critics, optimizers, and observation normalization statistics.

ARS checkpoints are saved to `checkpoints/<run_id>/ars_ckpt_<step>.pt` and include the linear policy weights, bias, and observation normalization statistics. ARS checkpoints are extremely lightweight.

DreamerV3 checkpoints are saved to `checkpoints/<run_id>/dreamer_ckpt_<step>.pt` and include the RSSM world model, latent actor, latent critic, optimizers, and observation normalization statistics. An actor-only checkpoint `dreamer_actor_<step>.pt` is also saved for deployment and race simulation.
