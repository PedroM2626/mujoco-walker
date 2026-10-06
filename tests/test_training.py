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


class TestSacUtdRatio(unittest.TestCase):
    """`--utd-ratio` makes the optimisation dose a setting instead of a consequence of `--num-envs`.

    With one gradient update per collection iteration, SAC applies its updates at
    1/`num_envs` per environment step: the 40M run at 32 envs optimised each environment step four
    times less than the same step budget at 8 envs, so the throughput flag was silently the dose
    flag. The ratio multiplies the updates per collection iteration, and 1 is the shipped schedule -
    which is what the committed runs were trained under, so nothing published moves.

    The counting is done by wrapping `ReplayBuffer.sample`: every critic update draws exactly one
    batch, so the draws are the gradient steps, and no new instrumentation had to be added to the
    trainer to observe its own schedule.
    """

    ROOT = os.path.dirname(os.path.dirname(__file__))
    RUN_ID = "utd_ratio_test"
    TOTAL = 1024
    LEARNING_STARTS = 128

    @classmethod
    def setUpClass(cls):
        import time
        from unittest import mock
        import train_walker

        cls.counts = {}
        real_sample = train_walker.ReplayBuffer.sample

        def counting_sample(self, batch_size):
            cls.draws.append(batch_size)
            return real_sample(self, batch_size)

        for name, envs, ratio in (("n2_r1", 2, 1), ("n2_r3", 2, 3), ("n6_r3", 6, 3)):
            cls.draws = []
            argv = ["train_walker.py", "--algo", "sac", "--device", "cpu", "--run-id", cls.RUN_ID,
                    "--seed", "7", "--total-timesteps", str(cls.TOTAL),
                    "--learning-starts", str(cls.LEARNING_STARTS), "--batch-size", "64",
                    "--buffer-size", "4096", "--num-envs", str(envs), "--utd-ratio", str(ratio)]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                    train_walker.ReplayBuffer, "sample", counting_sample):
                train_walker.train(start_time=time.time())
            cls.counts[name] = len(cls.draws)

    @classmethod
    def tearDownClass(cls):
        for path in (os.path.join("checkpoints", cls.RUN_ID),):
            if os.path.exists(path):
                shutil.rmtree(path)
        runs = os.path.join("runs")
        if os.path.exists(runs):
            for d in os.listdir(runs):
                if d.startswith(cls.RUN_ID):
                    shutil.rmtree(os.path.join(runs, d))

    def expected_updates(self, num_envs):
        """Collection iterations that reach the update block, at ratio 1."""
        return self.TOTAL // num_envs - self.LEARNING_STARTS // num_envs

    def test_the_ratio_multiplies_the_gradient_updates(self):
        one, three = self.counts["n2_r1"], self.counts["n2_r3"]
        self.assertGreater(one, 0, "no gradient update was observed; the counter is blind")
        self.assertEqual(one, self.expected_updates(2),
                         f"ratio 1 did one update per collection iteration, expected "
                         f"{self.expected_updates(2)} and saw {one}")
        self.assertEqual(three, 3 * one,
                         f"ratio 3 saw {three} updates against ratio 1's {one}")

    def test_the_dose_is_now_independent_of_the_environment_count(self):
        """n=2 at ratio 1 and n=6 at ratio 3 optimise the same number of environment steps alike.

        They cannot be exactly equal: the collection loop floors the iteration count, and
        `global_step > learning_starts` lands on a different step for each divisor. A handful of
        updates is the difference between 448 and 450, while the old coupling put a factor of three
        between the two.
        """
        self.assertLessEqual(abs(self.counts["n2_r1"] - self.counts["n6_r3"]), 8,
                             f"{self.counts['n2_r1']} against {self.counts['n6_r3']}: the dose still "
                             "follows --num-envs rather than --utd-ratio")
        self.assertAlmostEqual(self.counts["n6_r3"], 3 * self.expected_updates(6), delta=8,
                               msg="the n=6 run did not run three updates per collection iteration "
                                   "over the iterations its schedule reaches")

    def test_only_the_actor_critic_loops_got_the_loop_and_the_target_keeps_pace(self):
        """The ratio belongs to the two actor-critic schedules, and it must drag the targets along.

        A Polyak average is a lag measured in gradient steps, so applying it once per collection
        iteration at ratio 4 would leave the target four times further behind the critic than the
        shipped ratio-1 schedule does - a second, quieter semantics change on top of the dose. The
        loss logging is the opposite case: its cadence is the tensorboard series the training-rate
        summaries read, so it stays outside the loop and is untouched by the ratio.
        """
        lines = open(os.path.join(self.ROOT, "train_walker.py"), encoding="utf-8").read().splitlines()
        indent = lambda text: len(text) - len(text.lstrip())
        loops = [i for i, text in enumerate(lines) if "for _utd in range(max(1, args.utd_ratio))" in text]
        self.assertEqual(len(loops), 2, "SAC and TD3 should each own exactly one UTD loop")
        enclosing = set()
        for i in loops:
            enclosing.add(next(lines[j].split("(")[0].replace("def ", "")
                               for j in range(i, 0, -1) if lines[j].startswith("def ")))
        self.assertEqual(enclosing, {"train_td3", "train"},
                         f"the UTD loop appeared in {enclosing}; PPO has its own epochs x minibatches")
        for i in loops:
            body = indent(lines[i + 1])
            for j in range(i + 1, i + 80):
                if "args.target_network_frequency" in lines[j]:
                    self.assertEqual(indent(lines[j]), body,
                                     "the target sync left the UTD loop: tau would be applied per "
                                     "collection iteration while the critic moves per gradient step")
                    break
            else:
                self.fail(f"no target sync found after the loop at line {i + 1}")
            for j in range(i + 1, i + 80):
                if "if global_step % 1000 < args.num_envs:" in lines[j]:
                    self.assertLess(indent(lines[j]), body,
                                    "the loss logging moved inside the UTD loop, which multiplies "
                                    "the tensorboard cadence by the ratio")
                    break

    def test_ratio_one_is_the_default_and_the_flag_reads_it_like_every_other_knob(self):
        import train_walker
        self.assertEqual(train_walker.ENV_VARS["UTD_RATIO"], "1",
                         "the shipped schedule is one update per collection iteration; every "
                         "committed run was trained under it")
        source = open(os.path.join(self.ROOT, "train_walker.py"), encoding="utf-8").read()
        flag = source.index('"--utd-ratio"')
        self.assertIn('ENV_VARS["UTD_RATIO"]', source[flag:flag + 400],
                      "the flag does not take its default from ENV_VARS, so it would not follow the "
                      "environment-variable convention the rest of this parser uses")


if __name__ == "__main__":
    unittest.main()
