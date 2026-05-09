"""Integration test for short training run."""

import os
import sys
import unittest
import subprocess
import shutil
import torch

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
                "sac_walker.py",
                "--run-id", self.run_id,
                "--seed", "42",
                "--total-timesteps", "4096",
                "--num-envs", "2",
                "--learning-starts", "128",
                "--batch-size", "64",
                "--buffer-size", "4096",
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

    def test_resume_training_restores_checkpoint_state(self):
        first = subprocess.run(
            [
                sys.executable,
                "sac_walker.py",
                "--run-id", self.run_id,
                "--seed", "42",
                "--total-timesteps", "2048",
                "--num-envs", "2",
                "--learning-starts", "128",
                "--batch-size", "64",
                "--buffer-size", "4096",
            ],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            capture_output=True,
            text=True,
        )
        self.assertEqual(first.returncode, 0, f"Initial training failed: {first.stderr}")

        second = subprocess.run(
            [
                sys.executable,
                "sac_walker.py",
                "--run-id", self.run_id,
                "--seed", "42",
                "--resume",
                "--total-timesteps", "4096",
                "--num-envs", "2",
                "--learning-starts", "128",
                "--batch-size", "64",
                "--buffer-size", "4096",
            ],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            capture_output=True,
            text=True,
        )
        print(second.stdout)
        print(second.stderr)
        self.assertEqual(second.returncode, 0, f"Resume training failed: {second.stderr}")

        ckpt_path = os.path.join("checkpoints", self.run_id, "sac_ckpt_4096.pt")
        self.assertTrue(os.path.exists(ckpt_path), "Resumed checkpoint not created")

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        self.assertEqual(checkpoint["global_step"], 4096)
        self.assertIn("replay_buffer", checkpoint)
        self.assertGreaterEqual(checkpoint["replay_buffer"]["pos"], 0)
        self.assertIn("rng_state", checkpoint)


if __name__ == "__main__":
    unittest.main()
