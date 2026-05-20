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

# Train SAC for 1M steps from the 40M baseline, upright resets only,
# terminate on fall so the gradient for walking is clear and strong.
run_cmd(
    ".venv\\Scripts\\python.exe train_walker.py"
    " --algo sac"
    " --run-id walker_sac_1m_v2"
    " --task-phase target"
    " --init-from-run-id walker_target_v1"
    " --reset-mode upright"
    " --total-timesteps 1000000"
    " --checkpoint-interval 200000"
    " --actor-learning-starts 50000"
    " --no-autotune"
    " --learning-rate 3e-4"
    " --num-envs 16"
    " --no-save-replay-buffer"
    " --force"
)

print("\n==================================================")
print("SAC 1M v2 TRAINING COMPLETED.")
print("==================================================\n")
