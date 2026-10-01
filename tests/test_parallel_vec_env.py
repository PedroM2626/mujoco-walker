"""Parity tests for the process-parallel vector environment.

The parallel vec env is only safe as a default if a trainer cannot tell it apart from
SyncVectorEnv, so every assertion below compares the two directly.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym  # noqa: E402
from envs.parallel_vector_env import ParallelVectorEnv, _sync_vector_env_compatible  # noqa: E402
from train_walker import make_env  # noqa: E402

# The parallel backend subclasses the SyncVectorEnv hook API that gymnasium 1.0 removed.
SUPPORTED = _sync_vector_env_compatible()

NUM_ENVS = 4
# Long enough that TimeLimit truncation fires, so the autoreset path is covered.
N_STEPS = 1050


def _env_kwargs(task_phase, reset_mode="upright"):
    # Start standing, then terminate on unhealthy z: with the fallen start the env can
    # not become unhealthy at all (it starts there), so no episode would ever end and
    # the autoreset path would go untested.
    return {
        "reset_mode": reset_mode,
        "task_phase": task_phase,
        "target_forward_velocity": 0.8,
        "terminate_when_unhealthy": True,
    }


def _specs(run_name, task_phase, reset_mode="upright"):
    return [
        (
            "train_walker",
            "make_env",
            ("WalkerRagdoll-v0", i, False, run_name),
            _env_kwargs(task_phase, reset_mode),
        )
        for i in range(NUM_ENVS)
    ]


def _assert_infos_close(case, a, b, msg):
    """Compare two info dicts, tolerating numpy-scalar -> Python-scalar conversion."""
    case.assertEqual(sorted(a), sorted(b), f"{msg}: keys")
    for key in a:
        # RecordEpisodeStatistics stamps episodes with time.time(); not comparable.
        if key == "t" and msg.endswith("episode"):
            continue
        va, vb = a[key], b[key]
        if isinstance(va, dict) or isinstance(vb, dict):
            case.assertIsInstance(vb, dict, f"{msg}: {key} should be a dict")
            _assert_infos_close(case, va, vb, f"{msg}.{key}")
        elif isinstance(va, str) or isinstance(vb, str):
            case.assertEqual(va, vb, f"{msg}: {key}")
        else:
            case.assertAlmostEqual(float(va), float(vb), places=9, msg=f"{msg}: {key}")


class TestParallelMatchesSync(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SUPPORTED:
            raise unittest.SkipTest(
                f"gymnasium {gym.__version__} removed SyncVectorEnv.reset_wait/step_wait; "
                "the parallel backend targets gymnasium>=0.29.1,<1.0"
            )

    def _compare(self, task_phase="recovery", sparse_info=False):
        sync = gym.vector.SyncVectorEnv(
            [
                (lambda i=i: make_env(
                    "WalkerRagdoll-v0", i, False, "parity_sync", **_env_kwargs(task_phase)
                )())
                for i in range(NUM_ENVS)
            ]
        )
        par = ParallelVectorEnv(_specs("parity_par", task_phase), sparse_info=sparse_info)
        try:
            self.assertEqual(sync.observation_space.shape, par.observation_space.shape)
            self.assertEqual(sync.action_space.shape, par.action_space.shape)

            obs_sync, info_sync = sync.reset(seed=11)
            obs_par, info_par = par.reset(seed=11)
            np.testing.assert_allclose(obs_sync, obs_par, rtol=0, atol=1e-12)
            self.assertEqual(set(info_sync), set(info_par))

            rng = np.random.default_rng(3)
            actions = rng.uniform(-1.0, 1.0, size=(N_STEPS, NUM_ENVS, 17)).astype(np.float32)
            terminal_seen = 0
            for step in range(N_STEPS):
                a = actions[step]
                ns, rs, ts, cs, is_ = sync.step(a)
                npr, rp, tp, cp, ip = par.step(a)
                np.testing.assert_allclose(ns, npr, rtol=0, atol=1e-12, err_msg=f"obs @{step}")
                np.testing.assert_allclose(rs, rp, rtol=0, atol=1e-12, err_msg=f"reward @{step}")
                self.assertEqual(list(ts), list(tp), f"terminated @{step}")
                self.assertEqual(list(cs), list(cp), f"truncated @{step}")

                if not sparse_info:
                    self.assertEqual(sorted(is_), sorted(ip), f"info keys @{step}")
                    for key in is_:
                        value = is_[key]
                        if isinstance(value, np.ndarray) and value.dtype != object:
                            np.testing.assert_allclose(value, ip[key], rtol=0, atol=1e-12)

                if "final_observation" in is_:
                    terminal_seen += 1
                    self.assertIn("final_observation", ip, f"autoreset missing @{step}")
                    # Object arrays hold None for the envs that did not finish.
                    sync_final = is_["final_observation"]
                    par_final = ip["final_observation"]
                    present = [entry is not None for entry in sync_final]
                    self.assertEqual(
                        present,
                        [entry is not None for entry in par_final],
                        f"which envs auto-reset @{step}",
                    )
                    self.assertTrue(any(present), f"empty final_observation @{step}")
                    np.testing.assert_allclose(
                        np.stack([o for o in sync_final if o is not None]),
                        np.stack([o for o in par_final if o is not None]),
                        rtol=0,
                        atol=1e-12,
                        err_msg=f"final_observation @{step}",
                    )
                    sync_fi = [f for f in is_["final_info"] if f is not None]
                    par_fi = [f for f in ip["final_info"] if f is not None]
                    self.assertEqual(len(sync_fi), len(par_fi), f"final_info count @{step}")
                    for a_info, b_info in zip(sync_fi, par_fi):
                        _assert_infos_close(self, a_info, b_info, f"final_info @{step}")

            self.assertGreater(terminal_seen, 0, "no episode ended; autoreset path untested")
        finally:
            sync.close()
            par.close()

    def test_recovery_phase_matches_sync(self):
        self._compare("recovery")

    def test_target_phase_matches_sync(self):
        self._compare("target")

    def test_sparse_info_keeps_rewards_and_autoreset(self):
        """sparse_info drops the per-step info dict but must not change anything else."""
        self._compare("recovery", sparse_info=True)

    def test_spaces_and_metadata(self):
        env = ParallelVectorEnv(_specs("parity_meta", "recovery"))
        try:
            self.assertEqual(env.num_envs, NUM_ENVS)
            self.assertTrue(env.is_vector_env)
            self.assertEqual(env.single_observation_space.shape, (46,))
            obs, _ = env.reset(seed=0)
            self.assertEqual(obs.shape, (NUM_ENVS, 46))
        finally:
            env.close()

    def test_seed_offsets_are_per_env(self):
        """SyncVectorEnv expands seed=int into [seed, seed+1, ...]; keep that."""
        env = ParallelVectorEnv(_specs("parity_seed", "recovery"))
        try:
            first, _ = env.reset(seed=5)
            second, _ = env.reset(seed=5)
            np.testing.assert_allclose(first, second, rtol=0, atol=0)
        finally:
            env.close()

    def test_partial_termination_resets_only_done_envs(self):
        """The autoreset dispatch must address the finished envs, not the first N."""
        env = ParallelVectorEnv(_specs("parity_partial", "recovery"), sparse_info=True)
        try:
            env.reset(seed=1)
            rng = np.random.default_rng(11)
            partial = False
            for _ in range(600):
                a = rng.uniform(-1, 1, size=(NUM_ENVS, 17)).astype(np.float32)
                _, _, te, tr, infos = env.step(a)
                finished = np.logical_or(te, tr)
                if 0 < int(finished.sum()) < NUM_ENVS:
                    partial = True
                    self.assertEqual(
                        [o is not None for o in infos["final_observation"]],
                        [bool(flag) for flag in finished],
                    )
                    break
            self.assertTrue(partial, "no partially-terminal step occurred")
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
