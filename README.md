# Walker Ragdoll - SAC with CleanRL and MuJoCo

Walker ragdoll training project using SAC (Soft Actor-Critic) with MuJoCo and a CleanRL-style environment.

## Project Structure

```
.
├── sac_walker.py              # Main SAC training script
├── play.py                    # Real-time visualization of a trained agent
├── walker_ragdoll.xml         # MuJoCo model for the walker ragdoll
├── envs/
│   └── walker_ragdoll_env.py # Custom Gymnasium environment
├── utils/
│   └── checkpoint.py          # Checkpoint utilities (save, load, resume, force)
├── tests/
│   ├── test_env.py            # Environment unit tests
│   └── test_training.py       # Training integration tests
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

### Basic training
```bash
python sac_walker.py --run-id walker_experiment_1 --seed 1 --total-timesteps 1000000
```

### With environment variables
```bash
# Linux/Mac
export RUN_ID=walker_experiment_1
export TOTAL_TIMESTEPS=1000000
python sac_walker.py

# Windows PowerShell
$env:RUN_ID = "walker_experiment_1"
$env:TOTAL_TIMESTEPS = "1000000"
python sac_walker.py
```

### Unity ML-Agents-style features

- **Automatic checkpointing**: Saves checkpoints at each configured interval.
- **Resume**: Continue training from where it stopped.
- **Force**: Delete previous data and start from scratch.
- **Run ID**: Unique identifier for each experiment.

#### Examples

**Start a new training run:**
```bash
python sac_walker.py --run-id walker_v1 --seed 1 --total-timesteps 1000000
```

**Resume training:**
```bash
python sac_walker.py --run-id walker_v1 --resume
```

**Force restart (deletes previous checkpoints/logs):**
```bash
python sac_walker.py --run-id walker_v1 --force --seed 2
```

**Capture video:**
```bash
python sac_walker.py --run-id walker_v1 --capture-video --total-timesteps 100000
```

## Hyperparameters

All hyperparameters can be configured via command line or environment variables.

| CLI Parameter | Environment Variable | Default | Description |
|---|---|---|---|
| `--run-id` | `RUN_ID` | `walker_sac` | Experiment ID |
| `--seed` | `SEED` | `1` | Random seed |
| `--checkpoint-interval` | `CHECKPOINT_INTERVAL` | `1000000` | Checkpoint interval in timesteps |
| `--total-timesteps` | `TOTAL_TIMESTEPS` | `1000000` | Total timesteps |
| `--learning-rate` | `LEARNING_RATE` | `3e-4` | Learning rate |
| `--num-envs` | `NUM_ENVS` | `16` | Number of parallel environments |
| `--buffer-size` | `BUFFER_SIZE` | `1000000` | Replay buffer size |
| `--batch-size` | `BATCH_SIZE` | `256` | SAC batch size |
| `--learning-starts` | `LEARNING_STARTS` | `10000` | Random exploration steps before updates |
| `--gamma` | `GAMMA` | `0.99` | Discount factor |
| `--tau` | `TAU` | `0.005` | Target network update rate |
| `--alpha` | `ALPHA` | `0.2` | Initial entropy temperature |
| `--reset-mode` | `RESET_MODE` | `mixed` | Initial-state distribution: `fixed`, `mixed`, `fallen`, or `upright` |
| `--fixed-reset-probability` | `FIXED_RESET_PROBABILITY` | `0.25` | In mixed mode, fraction of episodes using the old fixed fallen pose |
| `--upright-reset-probability` | `UPRIGHT_RESET_PROBABILITY` | `0.15` | In mixed mode, fraction of episodes starting almost upright |
| `--fallen-velocity-scale` | `FALLEN_VELOCITY_SCALE` | `0.35` | Extra velocity noise for randomized fallen resets |
| `--task-phase` | `TASK_PHASE` | `recovery` | Reward curriculum phase: `recovery`, `balance`, `walk`, or `target` |
| `--target-forward-velocity` | `TARGET_FORWARD_VELOCITY` | `0.8` | Target x velocity for the walk phase |
| `--init-from-run-id` | `INIT_FROM_RUN_ID` | - | Start a new run from another SAC run's actor weights and observation normalization |
| `--init-from-checkpoint-step` | `INIT_FROM_CHECKPOINT_STEP` | `0` | Specific checkpoint step for `--init-from-run-id`; `0` means latest |
| `--mlflow-experiment` | `MLFLOW_EXPERIMENT` | `walker-ragdoll-sac` | MLflow experiment name |
| `--disable-mlflow` | - | `False` | Disable MLflow tracking |

## Training Recommendations

For the `WalkerRagdoll-v0` environment, the reward is split into phases. `recovery` keeps the stand-up objective, `balance` adds standing stability, foot-only support, low drift, and low torso velocity, `walk` adds gated forward-velocity tracking after the agent is upright, and `target` adds a sampled navigation target with reward for reducing distance while standing. Old PPO checkpoints are not compatible with this task definition.

The default reset distribution is `mixed`: some episodes keep the original fixed fallen pose, some start nearly upright, and the rest start from randomized fallen poses with different torso orientations, joint offsets, and velocity perturbations. This makes recovery training cover more of the states the agent reaches after real falls instead of overfitting to one spawn pose.

Useful reset experiments:

```bash
python sac_walker.py --run-id walker_recovery_v1 --force --total-timesteps 20000000
python sac_walker.py --run-id walker_recovery_v1 --resume --reset-mode fallen
python sac_walker.py --run-id walker_recovery_v1 --resume --fixed-reset-probability 0.10 --upright-reset-probability 0.20
```

Recommended phase progression after `walker_recovery_v1`:

```bash
# Phase 2: learn to stay upright for longer, without reusing old replay rewards.
python sac_walker.py --run-id walker_balance_v1 --task-phase balance --init-from-run-id walker_recovery_v1 --total-timesteps 30000000 --upright-reset-probability 0.35 --fixed-reset-probability 0.05

# Phase 3: start walking from the balanced policy.
python sac_walker.py --run-id walker_walk_v1 --task-phase walk --init-from-run-id walker_balance_v1 --total-timesteps 50000000 --upright-reset-probability 0.40 --fixed-reset-probability 0.05 --target-forward-velocity 0.8

# Phase 4: walk to sampled targets using the walking policy as initialization.
python sac_walker.py --run-id walker_target_v1 --task-phase target --init-from-run-id walker_walk_v1 --total-timesteps 70000000 --upright-reset-probability 0.40 --fixed-reset-probability 0.05 --target-forward-velocity 0.8
```

`--init-from-run-id` loads the actor and observation normalization from the source checkpoint, then starts a fresh replay buffer and optimizer state. This is the safest path when changing reward phases. For `target`, the observation grows from 46 to 49 dimensions; the loader copies the trained body-control input weights and initializes the new target inputs normally. Use `--resume` only to continue the same phase/run, and use `--init-critics` only for experiments where the reward and observation shape are very similar.

When using `--resume`, SAC restores the actor, critics, target networks, optimizers, entropy temperature, replay buffer, RNG state, and the active `NormalizeObservation` statistics. TensorBoard resumes into `runs/<run_id>__<seed>/`.

The environment reward is versioned. If the reward/contact logic changes, start a new SAC run with `--force`; resuming an older checkpoint would reuse replay-buffer rewards from the old objective.

## Tests

Run unit and integration tests:
```bash
python -m unittest discover -s tests -v
```

Or individually:
```bash
python -m tests.test_env
python -m tests.test_training
```

## TensorBoard

View training logs:
```bash
tensorboard --logdir runs/
```

## Visualizing the Trained Agent

Use the `play.py` script to watch the trained agent in real time.

### Watch live (opens a MuJoCo window, runs until you press Ctrl+C)
```bash
python play.py --run-id walker_v1 --render-mode human
```

### Run a fixed number of episodes
```bash
python play.py --run-id walker_v1 --num-episodes 10 --render-mode human
```

### Stochastic mode (sample from the SAC policy)
```bash
python play.py --run-id walker_v1 --stochastic --render-mode human
```

### Load a specific checkpoint step
```bash
python play.py --run-id walker_v1 --checkpoint-step 500000 --render-mode human
```

### Visualize a specific reset distribution
```bash
python play.py --run-id walker_v1 --reset-mode fallen --render-mode human
python play.py --run-id walker_v1 --reset-mode fixed --render-mode human
```

**Note:** `play.py` loads the saved observation normalization statistics and freezes them during evaluation.

### Watch an old PPO checkpoint
```bash
python play_ppo_legacy.py --run-id Humanoid_Curriculum_v1 --checkpoint-step 133801920 --render-mode human --deterministic
```

**Note:** `play_ppo_legacy.py` embeds the old 48-dimensional PPO environment and freezes the checkpoint's saved observation normalization statistics, so the funny legacy behavior can still be replayed without affecting the SAC environment.

## Checkpoints

SAC checkpoints are saved to `checkpoints/<run_id>/sac_ckpt_<step>.pt` and include the actor, critics, entropy temperature, optimizers, and observation normalization statistics.

