"""Batched REDQ ensemble: same function as the ModuleList it replaces, and checkpoint-compatible.

The claim this guards is narrow. `BatchedSoftQEnsemble` exists only to stop paying kernel-launch
time for N separate critics (the timings are measured, not asserted: `python
bench_redq_ensemble.py --isolated-only` writes them to benchmarks/redq_ensemble_ab.json), so what
must not change is the arithmetic and what must survive is every checkpoint written by either
implementation.
"""

import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_walker import SoftQNetwork, BatchedSoftQEnsemble  # noqa: E402
from train_redq import (  # noqa: E402
    ensemble_q, ensemble_state_dict, load_ensemble_state, make_ensemble, soft_update_ensemble)

OBS, ACT, N = 49, 17, 10


def pair(seed=0):
    torch.manual_seed(seed)
    loop = nn.ModuleList([SoftQNetwork(OBS, ACT) for _ in range(N)])
    torch.manual_seed(seed)
    batched = BatchedSoftQEnsemble(OBS, ACT, N)
    return loop, batched


class TestBatchedEnsembleEquivalence(unittest.TestCase):
    def setUp(self):
        self.loop, self.batched = pair()
        torch.manual_seed(3)
        self.obs = torch.randn(256, OBS)
        self.act = torch.randn(256, ACT)

    def test_forward_matches_to_float32_rounding(self):
        with torch.no_grad():
            a, b = ensemble_q(self.loop, self.obs, self.act), ensemble_q(self.batched, self.obs, self.act)
        self.assertEqual(a.shape, b.shape)
        rel = (a - b).abs().max().item() / a.abs().max().item()
        self.assertLess(rel, 1e-5, f"batched forward drifted {rel:.2e} relative")

    def test_gradients_match_to_float32_rounding(self):
        for ens in (self.loop, self.batched):
            for p in ens.parameters():
                p.grad = None
        ensemble_q(self.loop, self.obs, self.act).mean().backward()
        ensemble_q(self.batched, self.obs, self.act).mean().backward()
        for i in range(N):
            for layer, stacked in ((0, self.batched.w1), (2, self.batched.w2), (4, self.batched.w3)):
                ref = self.loop[i].net[layer].weight.grad
                got = stacked.grad[i].t()
                rel = (got - ref).abs().max().item() / ref.abs().max().item()
                self.assertLess(rel, 1e-5, f"grad of critic {i} layer {layer} drifted {rel:.2e}")

    def test_checkpoint_layout_is_shared_both_ways(self):
        keys = set(self.loop.state_dict())
        self.assertEqual(keys, set(ensemble_state_dict(self.batched)))
        fresh_batched = BatchedSoftQEnsemble(OBS, ACT, N)
        load_ensemble_state(fresh_batched, self.loop.state_dict())
        self.assertTrue(torch.allclose(fresh_batched.w1, self.batched.w1))
        fresh_loop = nn.ModuleList([SoftQNetwork(OBS, ACT) for _ in range(N)])
        load_ensemble_state(fresh_loop, ensemble_state_dict(self.batched))
        self.assertTrue(torch.allclose(fresh_loop[3].net[2].weight, self.loop[3].net[2].weight))

    def test_soft_update_matches_the_per_parameter_loop(self):
        """The six-kernel lerp and the 3N-kernel lerp must land on the same target weights."""
        q_loop, q_bat = pair(seed=5)
        t_loop, t_bat = pair(seed=6)
        with torch.no_grad():
            for p in q_loop.parameters():
                p.add_(0.1)                       # train the source a little
        load_ensemble_state(q_bat, ensemble_state_dict(q_loop))   # identical source both paths
        soft_update_ensemble(t_loop, q_loop, 0.005)
        soft_update_ensemble(t_bat, q_bat, 0.005)
        after_loop, after_bat = ensemble_state_dict(t_loop), ensemble_state_dict(t_bat)
        self.assertEqual(set(after_loop), set(after_bat))
        for k, tensor in after_bat.items():
            self.assertTrue(torch.allclose(tensor, after_loop[k], atol=1e-6),
                            f"{k} drifted: {(tensor - after_loop[k]).abs().max():.2e}")
        with torch.no_grad():                     # and it really did move toward the source
            self.assertGreater((after_bat["0.net.0.weight"] -
                                pair(seed=6)[1].w1[0].t()).abs().max().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
