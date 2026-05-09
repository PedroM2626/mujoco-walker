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

## Training Recommendations

For the `WalkerRagdoll-v0` environment, the reward now follows a HumanoidStandup-style objective: torso height per control step, minus control and impact costs, with a small upright/standing stabilizer. Old PPO checkpoints are not compatible with this task definition.

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

**Note:** `play.py` loads the saved observation normalization statistics and freezes them during evaluation.

### Watch an old PPO checkpoint
```bash
python play_ppo_legacy.py --run-id Humanoid_Curriculum_v1 --checkpoint-step 133801920 --render-mode human --deterministic
```

**Note:** `play_ppo_legacy.py` embeds the old 48-dimensional PPO environment and freezes the checkpoint's saved observation normalization statistics, so the funny legacy behavior can still be replayed without affecting the SAC environment.

## Checkpoints

SAC checkpoints are saved to `checkpoints/<run_id>/sac_ckpt_<step>.pt` and include the actor, critics, entropy temperature, optimizers, and observation normalization statistics.

