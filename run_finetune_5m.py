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

# Fine-tune SAC for 5M more steps from the 1M checkpoint - very low LR to stabilize gait
run_cmd(".venv\\Scripts\\python.exe train_walker.py --algo sac --run-id walker_sac_5m --task-phase target --init-from-run-id walker_sac_1m --reset-mode upright --total-timesteps 5000000 --checkpoint-interval 500000 --actor-learning-starts 0 --no-autotune --learning-rate 1e-5 --num-envs 16 --force")

# Fine-tune PPO for 5M more steps from the 1M checkpoint
run_cmd(".venv\\Scripts\\python.exe train_walker.py --algo ppo --run-id walker_ppo_5m --task-phase target --init-from-run-id walker_ppo_1m --reset-mode upright --total-timesteps 5000000 --checkpoint-interval 500000 --learning-rate 1e-5 --num-envs 16 --force")

# Fine-tune TD3 for 5M more steps from the 1M checkpoint
run_cmd(".venv\\Scripts\\python.exe train_walker.py --algo td3 --run-id walker_td3_5m --task-phase target --init-from-run-id walker_td3_1m --reset-mode upright --total-timesteps 5000000 --checkpoint-interval 500000 --actor-learning-starts 0 --learning-rate 1e-5 --num-envs 16 --force")

print("\n==================================================")
print("ALL 5M FINE-TUNING RUNS COMPLETED SUCCESSFULLY!")
print("==================================================\n")
