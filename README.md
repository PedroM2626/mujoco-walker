# Walker Ragdoll - PPO with CleanRL and MuJoCo

Walker ragdoll training project using PPO (Proximal Policy Optimization) with MuJoCo and a CleanRL-style environment.

## Project Structure

```
.
├── ppo_walker.py              # Main PPO training script
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
python ppo_walker.py --run-id walker_experiment_1 --seed 1 --total-timesteps 1000000
```

### With environment variables
```bash
# Linux/Mac
export RUN_ID=walker_experiment_1
export TOTAL_TIMESTEPS=1000000
python ppo_walker.py

# Windows PowerShell
$env:RUN_ID = "walker_experiment_1"
$env:TOTAL_TIMESTEPS = "1000000"
python ppo_walker.py
```

### Unity ML-Agents-style features

- **Automatic checkpointing**: Saves checkpoints at each configured interval.
- **Resume**: Continue training from where it stopped.
- **Force**: Delete previous data and start from scratch.
- **Run ID**: Unique identifier for each experiment.

#### Examples

**Start a new training run:**
```bash
python ppo_walker.py --run-id walker_v1 --seed 1 --total-timesteps 1000000
```

**Resume training:**
```bash
python ppo_walker.py --run-id walker_v1 --resume
```

**Force restart (deletes previous checkpoints/logs):**
```bash
python ppo_walker.py --run-id walker_v1 --force --seed 2
```

**Capture video:**
```bash
python ppo_walker.py --run-id walker_v1 --capture-video --total-timesteps 100000
```

**Track with Weights & Biases:**
```bash
python ppo_walker.py --run-id walker_v1 --track --wandb-project walker-ragdoll
```

## Hyperparameters

All hyperparameters can be configured via command line or environment variables.

| CLI Parameter | Environment Variable | Default | Description |
|---|---|---|---|
| `--run-id` | `RUN_ID` | `walker_ppo` | Experiment ID |
| `--seed` | `SEED` | `1` | Random seed |
| `--checkpoint-interval` | `CHECKPOINT_INTERVAL` | `100000` | Checkpoint interval in timesteps |
| `--total-timesteps` | `TOTAL_TIMESTEPS` | `1000000` | Total timesteps |
| `--learning-rate` | `LEARNING_RATE` | `3e-4` | Learning rate |
| `--num-envs` | `NUM_ENVS` | `8` | Number of parallel environments |
| `--num-steps` | `NUM_STEPS` | `2048` | Steps per rollout |
| `--gamma` | `GAMMA` | `0.99` | Discount factor |
| `--gae-lambda` | `GAE_LAMBDA` | `0.95` | GAE lambda |
| `--num-minibatches` | `NUM_MINIBATCHES` | `32` | Minibatches per update |
| `--update-epochs` | `UPDATE_EPOCHS` | `10` | Update epochs |
| `--clip-coef` | `CLIP_COEF` | `0.2` | PPO clipping coefficient |
| `--ent-coef` | `ENT_COEF` | `0.0` | Entropy coefficient |
| `--vf-coef` | `VF_COEF` | `0.5` | Value function coefficient |
| `--max-grad-norm` | `MAX_GRAD_NORM` | `0.5` | Max gradient norm for clipping |
| `--target-kl` | `TARGET_KL` | `0.01` | KL divergence target for early stopping |

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

### Deterministic mode (no action noise)
```bash
python play.py --run-id walker_v1 --deterministic --render-mode human
```

### Load a specific checkpoint step
```bash
python play.py --run-id walker_v1 --checkpoint-step 500000 --render-mode human
```

**Note:** `play.py` runs a short warm-up (200 random steps by default) to populate observation normalization statistics so the agent behaves consistently.

## Checkpoints

Checkpoints are saved to `checkpoints/<run_id>/`. The system automatically keeps the 3 most recent checkpoints.

To load a specific checkpoint:
```python
from utils.checkpoint import load_checkpoint
load_checkpoint(agent, optimizer, "walker_v1", global_step=500000)
```
