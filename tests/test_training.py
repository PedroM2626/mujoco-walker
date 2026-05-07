"""Integration test for short training run."""

import os
import sys
import unittest
import subprocess
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestTrainingIntegration(unittest.TestCase):
    def setUp(self):
        self.run_id = "integration_test_run"
        # Clean up before
        ckpt_path = os.path.join("checkpoints", self.run_id)
        if os.path.exists(ckpt_path):
            shutil.rmtree(ckpt_path)
        run_path = os.path.join("runs")
        if os.path.exists(run_path):
            for d in os.listdir(run_path):
                if d.startswith(self.run_id):
                    shutil.rmtree(os.path.join(run_path, d))

    def tearDown(self):
        ckpt_path = os.path.join("checkpoints", self.run_id)
        if os.path.exists(ckpt_path):
            shutil.rmtree(ckpt_path)
        run_path = os.path.join("runs")
        if os.path.exists(run_path):
            for d in os.listdir(run_path):
                if d.startswith(self.run_id):
                    shutil.rmtree(os.path.join(run_path, d))

    def test_short_training(self):
        result = subprocess.run(
            [
                sys.executable,
                "ppo_walker.py",
                "--run-id", self.run_id,
                "--seed", "42",
                "--total-timesteps", "4096",
                "--num-steps", "128",
                "--num-envs", "2",
                "--update-epochs", "2",
                "--num-minibatches", "2",
            ],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        print(result.stderr)
        self.assertEqual(result.returncode, 0, f"Training failed: {result.stderr}")
        ckpt_dir = os.path.join("checkpoints", self.run_id)
        self.assertTrue(os.path.exists(ckpt_dir), "Checkpoint directory not created")
        self.assertTrue(any(f.endswith(".pt") for f in os.listdir(ckpt_dir)), "No checkpoint files found")


if __name__ == "__main__":
    unittest.main()
