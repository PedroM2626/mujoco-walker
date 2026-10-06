"""Integration test for short training run."""

import os
import sys
import tempfile
import unittest
import subprocess
import shutil
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# A test that spawns a real trainer must not write into the repository's MLflow archive. Every
# tracking-URI consumer here honours MLFLOW_TRACKING_URI (utils/mlflow_uri.py), and subprocesses
# inherit the environment, so this one line covers both the in-process and the spawned trainers.
# Without it the archive had 102 active runs that were integration/smoke noise rather than
# research history, and one killed test had left a run in RUNNING state forever.
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///" + os.path.join(
    tempfile.gettempdir(), "test_mlruns_walker.db").replace(os.sep, "/"))


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
        ppo_ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith("ppo_ckpt_")]
        self.assertTrue(ppo_ckpts, "No PPO checkpoint files found")

        # A trainer that produces a checkpoint the scorers cannot read has produced nothing. PPO
        # stores its network under `agent_state_dict`, which the shared actor branch does not look
        # for, and the scorer's own comment used to claim otherwise.
        import eval_phase1
        from envs.walker_ragdoll_env import ENV_VERSION
        path = os.path.join(ckpt_dir, sorted(ppo_ckpts)[-1])
        label, phase, width, policy, reset, reward_info = eval_phase1.build_policy(path, "cpu")
        self.assertEqual(label, "ppo")
        action = policy(torch.zeros(width, dtype=torch.float64).numpy())
        self.assertEqual(action.shape, (17,), "the PPO policy must return one action per dim")
        # No bound asserted on the value: PPO's deterministic action is the raw mean head, and the
        # environment clips it, so a trained policy can and does hand back |a| > 1.
        self.assertEqual(reward_info["env_version"], ENV_VERSION)


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


class TestRedqBatchedEnsembleRuns(unittest.TestCase):
    """The real REDQ loop with the batched critics.

    tests/test_redq_ensemble.py proves the arithmetic; this proves the wiring, because the batched
    stack changes tensor shape in three places the unit test does not exercise: the M-of-N target
    index, the actor back-propagating through the ensemble mean, and the checkpoint round-trip.
    Pinned to CPU for the same reason as the other trainer subprocess tests - a CUDA-using test
    subprocess competes with whatever is training on the machine.
    """

    def setUp(self):
        self.run_id = "integration_test_redq"
        for path in (os.path.join("checkpoints", self.run_id),):
            if os.path.exists(path):
                shutil.rmtree(path)

    def tearDown(self):
        path = os.path.join("checkpoints", self.run_id)
        if os.path.exists(path):
            shutil.rmtree(path)
        runs = os.path.join("runs")
        if os.path.isdir(runs):
            for d in os.listdir(runs):
                if d.startswith(self.run_id):
                    shutil.rmtree(os.path.join(runs, d))

    def _train(self, impl, extra=()):
        return subprocess.run(
            [sys.executable, "train_redq.py", "--run-id", self.run_id, "--seed", "42",
             "--device", "cpu", "--vec-backend", "sync", "--num-envs", "2",
             "--total-timesteps", "512", "--learning-starts", "128", "--batch-size", "32",
             "--buffer-size", "512", "--ensemble-size", "3", "--num-min-critics", "2",
             "--utd-ratio", "2", "--policy-frequency", "2", "--checkpoint-interval", "1000",
             "--ensemble-impl", impl, "--task-phase", "target", *extra],
            cwd=os.path.dirname(os.path.dirname(__file__)), capture_output=True, text=True)

    def test_short_redq_training_with_batched_ensemble(self):
        result = self._train("batched")
        self.assertEqual(result.returncode, 0, f"batched REDQ failed: {result.stderr[-1500:]}")
        files = os.listdir(os.path.join("checkpoints", self.run_id))
        self.assertTrue(any(f.startswith("redq_ckpt_") for f in files), files)

    def test_batched_checkpoint_resumes(self):
        self.assertEqual(self._train("batched").returncode, 0)
        again = self._train("batched", extra=["--resume"])
        self.assertEqual(again.returncode, 0,
                         "resuming a batched checkpoint into the batched ensemble failed: "
                         + again.stderr[-1500:])

    def test_switching_ensemble_impl_on_resume_is_refused_with_a_reason(self):
        """Adam state is per parameter, so the layout that wrote it has to resume it.

        Weights do cross over - that is what --init-from-run-id is for - but silently rebuilding
        an optimizer over a different parameter set would quietly change the run, so the trainer
        says no instead.
        """
        self.assertEqual(self._train("loop").returncode, 0)
        cross = self._train("batched", extra=["--resume"])
        self.assertNotEqual(cross.returncode, 0, "cross-impl resume should not be allowed")
        self.assertIn("ensemble-impl", cross.stderr + cross.stdout)


if __name__ == "__main__":
    unittest.main()
