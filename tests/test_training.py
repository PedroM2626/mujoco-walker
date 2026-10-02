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
                "train_walker.py",
                # Pinned to CPU: these tests spawn real trainers and must not contend for the
                # GPU, which on this laptop is also where the long runs live. Under a running
                # training job a CUDA-using test subprocess failed once with
                # "unspecified launch failure".
                "--device", "cpu",
                "--run-id", self.run_id,
                "--seed", "42",
                "--total-timesteps", "1024",
                "--num-envs", "2",
                "--learning-starts", "128",
                "--batch-size", "64",
                "--buffer-size", "1024",
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
                "train_walker.py",
                "--device", "cpu",
                "--run-id", self.run_id,
                "--seed", "42",
                "--total-timesteps", "512",
                "--num-envs", "2",
                "--learning-starts", "128",
                "--batch-size", "64",
                "--buffer-size", "1024",
            ],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            capture_output=True,
            text=True,
        )
        self.assertEqual(first.returncode, 0, f"Initial training failed: {first.stderr}")

        second = subprocess.run(
            [
                sys.executable,
                "train_walker.py",
                "--device", "cpu",
                "--run-id", self.run_id,
                "--seed", "42",
                "--resume",
                "--total-timesteps", "1024",
                "--num-envs", "2",
                "--learning-starts", "128",
                "--batch-size", "64",
                "--buffer-size", "1024",
            ],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            capture_output=True,
            text=True,
        )
        print(second.stdout)
        print(second.stderr)
        self.assertEqual(second.returncode, 0, f"Resume training failed: {second.stderr}")

        ckpt_path = os.path.join("checkpoints", self.run_id, "sac_ckpt_1024.pt")
        self.assertTrue(os.path.exists(ckpt_path), "Resumed checkpoint not created")

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.assertEqual(checkpoint["global_step"], 1024)
        self.assertIn("replay_buffer", checkpoint)
        self.assertGreaterEqual(checkpoint["replay_buffer"]["pos"], 0)
        self.assertIn("rng_state", checkpoint)

        run_dir = os.path.join("runs", f"{self.run_id}__42")
        self.assertTrue(os.path.exists(run_dir), "Stable TensorBoard run directory not found")

    def test_short_ppo_training(self):
        result = subprocess.run(
            [
                sys.executable,
                "train_walker.py",
                "--device", "cpu",
                "--algo", "ppo",
                "--run-id", self.run_id,
                "--seed", "42",
                "--total-timesteps", "512",
                "--num-envs", "2",
                "--num-steps", "64",
                "--num-minibatches", "2",
                "--update-epochs", "2",
            ],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        print(result.stderr)
        self.assertEqual(result.returncode, 0, f"PPO training failed: {result.stderr}")
        ckpt_dir = os.path.join("checkpoints", self.run_id)
        self.assertTrue(os.path.exists(ckpt_dir), "Checkpoint directory not created")
        self.assertTrue(any(f.startswith("ppo_ckpt_") for f in os.listdir(ckpt_dir)), "No PPO checkpoint files found")


    def test_short_td3_training(self):
        result = subprocess.run(
            [
                sys.executable,
                "train_walker.py",
                "--device", "cpu",
                "--algo", "td3",
                "--run-id", self.run_id,
                "--seed", "42",
                "--total-timesteps", "512",
                "--num-envs", "2",
                "--learning-starts", "128",
                "--batch-size", "64",
                "--buffer-size", "1024",
            ],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        print(result.stderr)
        self.assertEqual(result.returncode, 0, f"TD3 training failed: {result.stderr}")
        ckpt_dir = os.path.join("checkpoints", self.run_id)
        self.assertTrue(os.path.exists(ckpt_dir), "Checkpoint directory not created")
        self.assertTrue(any(f.startswith("td3_ckpt_") for f in os.listdir(ckpt_dir)), "No TD3 checkpoint files found")


if __name__ == "__main__":
    unittest.main()
