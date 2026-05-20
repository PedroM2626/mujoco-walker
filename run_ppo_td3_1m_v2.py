import subprocess
import sys

def run_cmd(cmd):
    print(f"\n==================================================")
    print(f"STARTING: {cmd}")
    print(f"==================================================\n")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"Failed to run: {cmd}")
        sys.exit(res.returncode)

# Train PPO for 1M steps using same reward setup that made SAC work
run_cmd(
    ".venv\\Scripts\\python.exe train_walker.py"
    " --algo ppo"
    " --run-id walker_ppo_1m_v2"
    " --task-phase target"
    " --init-from-run-id walker_target_v1"
    " --reset-mode upright"
    " --total-timesteps 1000000"
    " --checkpoint-interval 200000"
    " --learning-rate 3e-4"
    " --num-envs 16"
    " --force"
)

# Train TD3 for 1M steps using same reward setup
run_cmd(
    ".venv\\Scripts\\python.exe train_walker.py"
    " --algo td3"
    " --run-id walker_td3_1m_v2"
    " --task-phase target"
    " --init-from-run-id walker_target_v1"
    " --reset-mode upright"
    " --total-timesteps 1000000"
    " --checkpoint-interval 200000"
    " --actor-learning-starts 50000"
    " --learning-rate 3e-4"
    " --num-envs 16"
    " --no-save-replay-buffer"
    " --force"
)

print("\n==================================================")
print("PPO + TD3 1M v2 TRAINING COMPLETED.")
print("==================================================\n")
