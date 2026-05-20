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

# 1. Train SAC for 1M steps
run_cmd(".venv\\Scripts\\python.exe train_walker.py --algo sac --run-id walker_sac_1m --task-phase target --init-from-run-id walker_target_v1 --reset-mode upright --total-timesteps 1000000 --checkpoint-interval 200000 --actor-learning-starts 100000 --no-autotune --learning-rate 5e-5 --num-envs 16 --force")

# 2. Train PPO for 1M steps
run_cmd(".venv\\Scripts\\python.exe train_walker.py --algo ppo --run-id walker_ppo_1m --task-phase target --init-from-run-id walker_target_v1 --reset-mode upright --total-timesteps 1000000 --checkpoint-interval 200000 --learning-rate 5e-5 --num-envs 16 --force")

# 3. Train TD3 for 1M steps
run_cmd(".venv\\Scripts\\python.exe train_walker.py --algo td3 --run-id walker_td3_1m --task-phase target --init-from-run-id walker_target_v1 --reset-mode upright --total-timesteps 1000000 --checkpoint-interval 200000 --actor-learning-starts 100000 --learning-rate 5e-5 --num-envs 16 --force")

print("\n==================================================")
print("ALL TRAINING PIPELINE RUNS COMPLETED SUCCESSFULLY!")
print("==================================================\n")
